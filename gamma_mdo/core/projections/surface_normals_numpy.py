"""Analytic incident-face normals for collapsed B-spline parameter strips and poles.

These helpers use the CAD control net and polynomial spans, not the projection
seed triangulation. A missing or ill-conditioned incident face is reported to
the caller instead of supplying a direction made from roundoff.
"""

import numpy as np

from lsdo_function_spaces.core.spaces.non_cython_bsplines.compute_basis_matrix_numpy_factory import (
    compute_basis_stencil_numpy,
)


class _PatchNormals:
    """Evaluate one coefficient state's local normal support and limit normals."""

    def __init__(self, info, coefficients):
        self.info = info
        self.coefficients = coefficients
        self.flat = coefficients.reshape(-1, coefficients.shape[-1])
        self.spans = [np.array([j for j in range(p, coefficients.shape[a]) if k[j] < k[j+1]])
                      for a, (p, k) in enumerate(zip(info.degrees, info.knot_vectors))]
        self.intervals = {}
        self.poles = {}
        self.pole_integrals = {}
        for axis in (0, 1):
            degree = info.degrees[axis]
            knots = info.knot_vectors[axis]
            for end, index in ((0., 0), (1., -1)):
                row = np.take(coefficients, index, axis=axis)
                clamped = knots[:degree+1] if end == 0 else knots[-degree-1:]
                if np.all(clamped == end) and np.all(row == row[:1]):
                    self.poles[axis, end] = row[0].copy()

    def pole(self, uv):
        """Identify an exactly collapsed clamped end row at the selected UV."""
        return next((key for key in self.poles if uv[key[0]] == key[1]), None)

    def pole_normal(self, axis, end):
        """Integrate the oriented normal fan from analytic radial mixed derivatives.

        With inward radial coordinate t, S = P + t**k q(v) + higher terms.
        The normal fan weighted by angular arc length is proportional to
        q cross q_v / |q|**2. The first nonzero radial derivative supplies q;
        its factorial and radial scale cancel. The sign accounts for which
        endpoint and parameter direction collapses. Knot-span quadrature
        integrates CAD derivatives, independently of the warm-start mesh and
        the arbitrary circumferential coordinate assigned to the pole.
        """
        key = (axis, end)
        if key in self.pole_integrals:
            return self.pole_integrals[key]
        other = 1-axis
        knots = self.info.knot_vectors[other]
        spans = [(knots[j], knots[j+1]) for j in self.spans[other]]
        radial_knots = self.info.knot_vectors[axis]
        j = self.spans[axis][0 if end == 0 else -1]
        width = radial_knots[j+1]-radial_knots[j]
        extent = max(float(np.max(np.ptp(self.flat, axis=0))), np.finfo(float).tiny)
        previous, stable, result = None, 0, None
        for count in (4, 8, 16, 32, 64):
            nodes, weights = np.polynomial.legendre.leggauss(count)
            positions = np.concatenate([a+(nodes+1)*(b-a)/2 for a, b in spans])
            weights = np.concatenate([weights*(b-a)/2 for a, b in spans])
            uv = np.empty((len(positions), 2))
            uv[:, axis], uv[:, other] = end, positions
            q = np.zeros((len(uv), 3))
            dq = np.zeros_like(q)
            found = np.zeros(len(uv), dtype=bool)
            for order in range(1, self.info.degrees[axis]+1):
                radial = [0, 0]
                radial[axis] = order
                value, error = self.partials(uv, tuple(radial))
                active = ~found & (np.linalg.norm(value, axis=1) > np.maximum(
                    error, 256*np.finfo(float).eps*extent/width**order))
                if np.any(active):
                    radial[other] = 1
                    mixed, _ = self.partials(uv, tuple(radial))
                    q[active], dq[active] = value[active], mixed[active]
                    found |= active
            if not np.all(found):
                break
            cross = np.cross(q, dq)/np.einsum('ij,ij->i', q, q)[:, None]
            orientation = (1 if end == 0 else -1)*(1 if axis == 0 else -1)
            integral = orientation*np.einsum('i,ij->j', weights, cross)
            total = float(weights @ np.linalg.norm(cross, axis=1))
            if total == 0 or np.linalg.norm(integral) <= 256*np.finfo(float).eps*total:
                break
            if previous is not None and np.linalg.norm(integral-previous) <= 1e-8*total:
                stable += 1
            else:
                stable = 0
            if stable == 2:
                result = (integral, total)
                break
            previous = integral
        self.pole_integrals[key] = result
        return result

    def partials(self, uv, orders):
        """Evaluate centered derivative stencils and roundoff envelopes in a batch."""
        columns, weights, _ = compute_basis_stencil_numpy(
            uv, self.info.degrees, self.info.knot_vectors,
            der_orders=orders, cache=self.info.space_cache,
        )
        data = self.flat[columns]
        terms = weights[:, :, None]*(data-data[:, :1])
        value = terms.sum(axis=1)
        if orders == (0, 0):
            value += data[:, 0]
        error = 64*np.finfo(float).eps*np.linalg.norm(np.abs(terms).sum(axis=1), axis=1)
        return value, error

    def span(self, uv, axis):
        """Locate the active polynomial span, including the final endpoint."""
        k = self.info.knot_vectors[axis]
        return int(np.clip(np.searchsorted(k, uv[axis], side='right')-1,
                           self.info.degrees[axis], self.coefficients.shape[axis]-1))

    def interval(self, uv, axis):
        """Find a maximal exactly constant strip on the active transverse support."""
        spans = [self.span(uv, a) for a in (0, 1)]
        key = (*spans, axis)
        if key in self.intervals:
            return self.intervals[key]
        other = 1-axis
        net = np.moveaxis(self.coefficients, axis, 0)
        transverse = slice(spans[other]-self.info.degrees[other], spans[other]+1)
        degree = self.info.degrees[axis]

        def constant(j):
            support = net[j-degree:j+1, transverse]
            return support[0] if np.all(support == support[:1]) else None

        value = constant(spans[axis])
        result = None
        if value is not None:
            valid = self.spans[axis]
            index = int(np.searchsorted(valid, spans[axis]))
            lo = hi = index
            while lo > 0 and np.array_equal(constant(valid[lo-1]), value):
                lo -= 1
            while hi+1 < len(valid) and np.array_equal(constant(valid[hi+1]), value):
                hi += 1
            knots = self.info.knot_vectors[axis]
            result = (float(knots[valid[lo]]), float(knots[valid[hi]+1]))
        self.intervals[key] = result
        return result

    def partial(self, uv, orders):
        """Evaluate a centered stencil and its floating-point summation envelope."""
        columns, weights, _ = compute_basis_stencil_numpy(
            np.asarray(uv).reshape(1, 2), self.info.degrees, self.info.knot_vectors,
            der_orders=orders, cache=self.info.space_cache,
        )
        data = self.flat[columns[0]]
        centered = data-data[0]
        terms = weights[0, :, None]*centered
        value = terms.sum(axis=0)
        if orders == (0, 0):
            value = value+data[0]
        error = 64*np.finfo(float).eps*np.linalg.norm(np.abs(terms).sum(axis=0))
        return value, error

    @staticmethod
    def face(tu, tv, sides=()):
        """Return the oriented unit normal and the physical sector angle."""
        nu, nv = np.linalg.norm(tu), np.linalg.norm(tv)
        if nu == 0 or nv == 0:
            return None
        a, b = tu/nu, tv/nv
        cross = np.cross(a, b)
        size = np.linalg.norm(cross)
        if size <= 256*np.finfo(float).eps:
            return None
        weight = np.pi
        sides = dict(sides)
        if len(sides) == 2:
            weight = np.arctan2(size, np.dot(a*sides[0], b*sides[1]))
        return cross/size, float(weight)

    def normal(self, uv, sides=()):
        """Return a regular face normal only when both tangents beat roundoff."""
        tu, eu = self.partial(uv, (1, 0))
        tv, ev = self.partial(uv, (0, 1))
        if np.linalg.norm(tu) <= eu or np.linalg.norm(tv) <= ev:
            return None
        return self.face(tu, tv, sides)

    def frames(self, uv):
        """Batch regular normals and unit-tangent angles with centered stencils."""
        tangents, valid = [], np.ones(len(uv), dtype=bool)
        for orders in ((1, 0), (0, 1)):
            columns, weights, _ = compute_basis_stencil_numpy(
                uv, self.info.degrees, self.info.knot_vectors,
                der_orders=orders, cache=self.info.space_cache,
            )
            data = self.flat[columns]
            terms = weights[:, :, None]*(data-data[:, :1])
            tangent = terms.sum(axis=1)
            length = np.linalg.norm(tangent, axis=1)
            error = 64*np.finfo(float).eps*np.linalg.norm(np.abs(terms).sum(axis=1), axis=1)
            valid &= length > error
            tangents.append(tangent/np.where(length > 0, length, 1)[:, None])
        cross = np.cross(*tangents)
        sine = np.linalg.norm(cross, axis=1)
        valid &= sine > 256*np.finfo(float).eps
        normals = cross/np.where(sine > 0, sine, 1)[:, None]
        cosine = np.einsum('ij,ij->i', *tangents)
        return normals, sine, cosine, valid

    def limit(self, uv, axis, boundary, side, point, sides=(), check=True):
        """Recover a one-sided normal using the first nonzero analytic derivative."""
        evaluation = np.asarray(uv, dtype=float).copy()
        evaluation[axis] = np.nextafter(boundary, np.inf if side > 0 else -np.inf)
        j = self.span(evaluation, axis)
        knots = self.info.knot_vectors[axis]
        width = knots[j+1]-knots[j]
        if width <= 0:
            return None
        projected, _ = self.partial(evaluation, (0, 0))
        proximity = 1e-9*max(1., np.max(np.abs(self.coefficients)))
        if np.linalg.norm(projected-point) > proximity:
            return None
        other = 1-axis
        orders = [0, 0]; orders[other] = 1
        transverse, error = self.partial(evaluation, tuple(orders))
        if np.linalg.norm(transverse) <= error:
            return None
        extent = max(float(np.max(np.ptp(self.flat, axis=0))), np.finfo(float).tiny)
        for order in range(1, self.info.degrees[axis]+1):
            orders = [0, 0]; orders[axis] = order
            tangent, error = self.partial(evaluation, tuple(orders))
            threshold = max(error, 256*np.finfo(float).eps*extent/width**order)
            if np.linalg.norm(tangent) <= threshold:
                continue
            tangent *= side**(order-1)
            tangents = [None, None]
            tangents[axis], tangents[other] = tangent, transverse
            result = self.face(*tangents, sides=sides)
            if result is None:
                return None
            if check:
                # Verify approach from the same regular span. Secant checks
                # remain independent regressions rather than SDF inputs.
                stable = 0
                for fraction in (1e-3, 1e-4, 1e-5, 1e-6):
                    offset = evaluation.copy(); offset[axis] = boundary+side*fraction*width
                    trial = self.normal(offset)
                    if trial is not None and np.dot(trial[0], result[0]) > np.cos(1e-3):
                        stable += 1
                    if stable == 2:
                        break
                if stable < 2:
                    return None
            return result
        return None
