import json
import numpy as np
from scipy.integrate import dblquad

MU_0 = 4.0 * np.pi * 1e-7


class PlanarPolygonCoil:
    def __init__(self, ns, nt, s, r_i, w=0.0):
        if ns < 3:
            raise ValueError(f"Number of sides ns must be >= 3, got {ns}")
        if nt < 1:
            raise ValueError(f"Number of turns nt must be >= 1, got {nt}")
        if r_i <= 0:
            raise ValueError(f"Inner radius r_i must be > 0, got {r_i}")
        if s <= 0:
            raise ValueError(f"Spacing s must be > 0, got {s}")

        self.ns = int(ns)
        self.nt = int(nt)
        self.s = float(s)
        self.r_i = float(r_i)
        self.w = float(w)
        self.pitch = self.s + self.w
        self.r_o = self.r_i + (self.nt - 1) * self.pitch

    def get_turn_radii(self):
        indices = np.arange(self.nt)
        radii_mm = self.r_i + indices * self.pitch
        return radii_mm * 1e-3

    def get_turn_segments(self, turn_idx):
        r = (self.r_i + turn_idx * self.pitch) * 1e-3
        m = np.arange(1, self.ns + 1)
        theta = 2.0 * (m - 1) * np.pi / self.ns
        starts = np.stack(
            [r * np.cos(theta), r * np.sin(theta), np.zeros(self.ns)], axis=1
        )
        ends = np.roll(starts, -1, axis=0)
        return starts, ends

    def get_all_segments(self):
        starts_list = []
        ends_list = []
        turn_idx_list = []
        side_idx_list = []

        for turn in range(self.nt):
            s, e = self.get_turn_segments(turn)
            starts_list.append(s)
            ends_list.append(e)
            turn_idx_list.append(np.full(self.ns, turn))
            side_idx_list.append(np.arange(1, self.ns + 1))

        return (
            np.concatenate(starts_list, axis=0),
            np.concatenate(ends_list, axis=0),
            np.concatenate(turn_idx_list, axis=0),
            np.concatenate(side_idx_list, axis=0),
        )


def build_rotation_matrix(phi, theta, psi):
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(phi), -np.sin(phi)],
            [0.0, np.sin(phi), np.cos(phi)],
        ]
    )
    ry = np.array(
        [
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ]
    )
    rz = np.array(
        [
            [np.cos(psi), -np.sin(psi), 0.0],
            [np.sin(psi), np.cos(psi), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return rx @ ry @ rz


class PartialMutualInductanceCalculator:
    @staticmethod
    def calculate_mutual_inductance(
        tx,
        rx,
        translation,
        rotation=(0.0, 0.0, 0.0),
        n_quad=10,
        return_turn_matrix=False,
    ):
        trans_m = np.array(translation, dtype=np.float64) * 1e-3
        phi, theta, psi = rotation
        rot_mat = build_rotation_matrix(phi, theta, psi)

        tx_starts, tx_ends, tx_turn_idx, _ = tx.get_all_segments()
        l_tx = tx_ends - tx_starts

        rx_starts_local, rx_ends_local, rx_turn_idx, _ = rx.get_all_segments()
        rx_starts = (rot_mat @ rx_starts_local.T).T + trans_m
        rx_ends = (rot_mat @ rx_ends_local.T).T + trans_m
        l_rx = rx_ends - rx_starts

        dot_prod = np.dot(l_tx, l_rx.T)

        nodes, weights = np.polynomial.legendre.leggauss(n_quad)
        s_pts = 0.5 * (nodes + 1.0)
        w_pts = 0.5 * weights
        w_mat = np.outer(w_pts, w_pts)

        pts_tx = tx_starts[:, None, :] + s_pts[None, :, None] * l_tx[:, None, :]
        pts_rx = rx_starts[:, None, :] + s_pts[None, :, None] * l_rx[:, None, :]

        diff = pts_tx[:, None, :, None, :] - pts_rx[None, :, None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        inv_dist = 1.0 / dist

        segment_integrals = np.einsum("ijkl,kl->ij", inv_dist, w_mat)

        factor = MU_0 / (4.0 * np.pi)
        m_segment_pairs = factor * dot_prod * segment_integrals

        total_m_h = float(np.sum(m_segment_pairs))
        total_m_nh = total_m_h * 1e9
        total_m_uh = total_m_h * 1e6

        result = {
            "M_H": total_m_h,
            "M_nH": total_m_nh,
            "M_uH": total_m_uh,
            "num_segment_pairs": tx_starts.shape[0] * rx_starts.shape[0],
            "tx_total_turns": tx.nt,
            "rx_total_turns": rx.nt,
        }

        if return_turn_matrix:
            turn_matrix = np.zeros((tx.nt, rx.nt), dtype=np.float64)
            for t_turn in range(tx.nt):
                mask_tx = tx_turn_idx == t_turn
                for r_turn in range(rx.nt):
                    mask_rx = rx_turn_idx == r_turn
                    turn_matrix[t_turn, r_turn] = (
                        np.sum(m_segment_pairs[mask_tx][:, mask_rx]) * 1e9
                    )
            result["turn_matrix_nH"] = turn_matrix

        return result

    @staticmethod
    def calculate_mutual_inductance_equation_1b(
        tx, rx, translation, rotation=(0.0, 0.0, 0.0), eps=1e-12
    ):
        trans_m = np.array(translation, dtype=np.float64) * 1e-3
        phi, theta, psi = rotation
        rot_mat = build_rotation_matrix(phi, theta, psi)

        total_m_h = 0.0
        factor = MU_0 / (4.0 * np.pi)

        for i in range(tx.nt):
            tx_starts, tx_ends = tx.get_turn_segments(i)
            for j in range(rx.nt):
                rx_starts_local, rx_ends_local = rx.get_turn_segments(j)
                rx_starts = (rot_mat @ rx_starts_local.T).T + trans_m
                rx_ends = (rot_mat @ rx_ends_local.T).T + trans_m

                for m_idx in range(tx.ns):
                    m = m_idx + 1
                    a1 = tx_starts[m_idx]
                    a2 = tx_ends[m_idx]

                    dx_t = a2[0] - a1[0]
                    cot_val = 1.0 / np.tan(np.pi * (2 * m - 1) / tx.ns)
                    delta4 = -cot_val

                    for k_idx in range(rx.ns):
                        b1 = rx_starts[k_idx]
                        b2 = rx_ends[k_idx]

                        delta1 = b2[0] - b1[0]
                        delta2 = b2[1] - b1[1]
                        delta3 = b2[2] - b1[2]

                        if abs(delta1) < eps or abs(dx_t) < eps:
                            l1 = a2 - a1
                            l2 = b2 - b1
                            dot_val = np.dot(l1, l2)
                            nodes, weights = np.polynomial.legendre.leggauss(8)
                            s = 0.5 * (nodes + 1.0)
                            w = 0.5 * weights
                            p1 = a1[:, None] + l1[:, None] * s[None, :]
                            p2 = b1[:, None] + l2[:, None] * s[None, :]
                            diff = p1[:, :, None] - p2[:, None, :]
                            dist = np.linalg.norm(diff, axis=0)
                            int_val = np.sum(np.outer(w, w) / dist)
                            total_m_h += factor * dot_val * int_val
                        else:
                            delta = 1.0 + (delta2 / delta1) * delta4

                            def integrand(x_t, x_r):
                                y_r = (delta2 / delta1) * (x_r - b1[0]) + b1[1]
                                z_r = (delta3 / delta1) * (x_r - b1[0]) + b1[2]
                                y_t = delta4 * (x_t - a1[0]) + a1[1]
                                dist_sq = (
                                    (x_t - x_r) ** 2 + (y_t - y_r) ** 2 + z_r**2
                                )
                                return 1.0 / np.sqrt(dist_sq)

                            xr_min, xr_max = min(b1[0], b2[0]), max(b1[0], b2[0])
                            xt_min, xt_max = min(a1[0], a2[0]), max(a1[0], a2[0])

                            sign_t = 1.0 if a2[0] > a1[0] else -1.0
                            sign_r = 1.0 if b2[0] > b1[0] else -1.0

                            val, _ = dblquad(
                                integrand,
                                xr_min,
                                xr_max,
                                xt_min,
                                xt_max,
                                epsabs=1e-12,
                                epsrel=1e-6,
                            )
                            m_mkij = factor * delta * val * sign_t * sign_r
                            total_m_h += m_mkij

        return total_m_h * 1e9

    @staticmethod
    def calculate_self_inductance(coil, wire_radius_mm=0.5, n_quad=10):
        r_w = wire_radius_mm * 1e-3
        starts, ends, _, _ = coil.get_all_segments()
        num_segments = starts.shape[0]
        l_vec = ends - starts
        lengths = np.linalg.norm(l_vec, axis=1)

        factor_self = MU_0 / (2.0 * np.pi)
        l_segments = factor_self * lengths * (np.log(2.0 * lengths / r_w) - 1.0)
        total_l_self = np.sum(l_segments)

        factor_mut = MU_0 / (4.0 * np.pi)
        nodes, weights = np.polynomial.legendre.leggauss(n_quad)
        s_pts = 0.5 * (nodes + 1.0)
        w_pts = 0.5 * weights
        w_mat = np.outer(w_pts, w_pts)

        pts = starts[:, None, :] + s_pts[None, :, None] * l_vec[:, None, :]
        dot_prod = np.dot(l_vec, l_vec.T)

        diff = pts[:, None, :, None, :] - pts[None, :, None, :, :]
        dist = np.linalg.norm(diff, axis=-1)

        np.fill_diagonal(dot_prod, 0.0)

        for i in range(num_segments):
            dist[i, i, :, :] = 1.0

        inv_dist = 1.0 / dist
        for i in range(num_segments):
            inv_dist[i, i, :, :] = 0.0

        integral = np.einsum("ijkl,kl->ij", inv_dist, w_mat)
        total_mutual_segments = factor_mut * np.sum(dot_prod * integral)

        l_total_h = total_l_self + total_mutual_segments
        return {
            "L_H": float(l_total_h),
            "L_nH": float(l_total_h * 1e9),
            "L_uH": float(l_total_h * 1e6),
        }


def main():
    tx = PlanarPolygonCoil(ns=7, nt=10, s=2.0, r_i=55.0, w=0.68)
    rx = PlanarPolygonCoil(ns=7, nt=10, s=2.0, r_i=55.0, w=0.68)
    result = PartialMutualInductanceCalculator.calculate_mutual_inductance(
        tx=tx, rx=rx, translation=(0.0, 0.0, 70.0), rotation=(0.0, 0.0, 0.0)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
