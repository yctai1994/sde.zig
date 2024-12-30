/// Modified midpoint step.
/// At `xstart`, input the dependent variable vector `y` and its derivative vector `dydx`.
/// Also input is htot, the total step to be made, and nstep, the number of substeps to be used.
/// The output is returned as `yout`, which need not be a distinct array from `y`;
/// if it is distinct, however, then `y` and `dydx` are returned unmodified.
pub fn mmid(
    y: []f64,
    dydx: []f64,
    xstart: f64,
    nstep: usize,
    hstep: f64, // stepsize of this trip.
    yout: []f64,
    derivs: *const fn (x: f64, yn: []f64, yout: []f64) void,
    ym: []f64,
    yn: []f64,
) !void {
    debug.assert(y.len == dydx.len);
    debug.assert(y.len == yout.len);
    debug.assert(y.len == ym.len);
    debug.assert(y.len == yn.len);

    // First Step:
    for (0..y.len) |i| {
        ym[i] = y[i];
        yn[i] = y[i] + hstep * dydx[i];
    }

    var xnew: f64 = xstart + hstep;
    derivs(xnew, yn, yout); // Will use yout for temporary storage of derivatives.
    const hstep_doub: f64 = 2.0 * hstep;

    // General Step
    var swap: f64 = undefined;
    for (1..nstep) |_| {
        for (0..y.len) |i| {
            swap = ym[i] + hstep_doub * yout[i];
            ym[i] = yn[i];
            yn[i] = swap;
        }
        xnew += hstep;
        derivs(xnew, yn, yout);
    }

    // Last step.
    for (0..y.len) |i| {
        yout[i] = 0.5 * (ym[i] + yn[i] + hstep * yout[i]);
    }
}

const std = @import("std");
const debug = std.debug;
