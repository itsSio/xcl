//! End-to-end checks of emitted DXIL: the SDK validator accepts it, DXC
//! disassembles it, and D3D12 runs it on the GPU with the recompiler's
//! kernel ABI (u0 `io`, u1 `res` root UAVs).

use super::container::Stage;
use super::module::{BinOp, CastOp, Pred};
use super::shader::Shader;
use super::validate;
use crate::gpu::recomp::runner;

/// The program of `tests/data/dxil/dxc_cs_raw_uav.hlsl`, emitted directly:
/// `a = io[lane]; f = asfloat(a) * 2 + 1; if (a > 5) f = -f;
///  res[lane] = asuint(f) + WaveReadLaneAt(a, 3)` — with a real branch and phi
/// rather than a select, to exercise control flow.
fn sample_kernel(wave: u32) -> Vec<u8> {
    let mut s = Shader::new(Stage::Compute, (6, 6));
    let io = s.raw_uav(0, 0);
    let res = s.raw_uav(1, 0);
    let lane = s.thread_id_in_group(0);
    let two = s.m.i32c(2);
    let off = s.m.bin(BinOp::Shl, lane, two);
    let a = s.raw_load(io, off, 1)[0];
    let f32 = s.m.f32();
    let i32 = s.m.i32();
    let fa = s.m.bitcast(a, f32);
    let c2 = s.m.f32c(2.0f32.to_bits());
    let c1 = s.m.f32c(1.0f32.to_bits());
    let mul = s.m.fbin(BinOp::Mul, fa, c2, 0);
    let f = s.m.fbin(BinOp::Add, mul, c1, 0);
    let five = s.m.i32c(5);
    let big = s.m.cmp(Pred::UGt, a, five);
    let three = s.m.i32c(3);
    let other = s.wave_read_lane_at(a, three);

    let entry = s.m.current_block();
    let neg_b = s.m.new_block();
    let join = s.m.new_block();
    s.m.cond_br(big, neg_b, join);
    s.m.set_block(neg_b);
    let nz = s.m.f32c(0x8000_0000);
    let neg = s.m.fbin(BinOp::Sub, nz, f, 0);
    s.m.br(join);
    s.m.set_block(join);
    let phi = s.m.phi(f32);
    s.m.add_incoming(phi, f, entry);
    s.m.add_incoming(phi, neg, neg_b);
    let bits = s.m.cast(CastOp::BitCast, phi, i32);
    let sum = s.m.bin(BinOp::Add, bits, other);
    s.raw_store(res, off, &[sum]);
    s.finish([64, 1, 1], Some(wave))
}

fn expected(io: &[u32], wave: usize) -> Vec<u32> {
    (0..64)
        .map(|l| {
            let a = io[l];
            let mut f = f32::from_bits(a) * 2.0 + 1.0;
            if a > 5 {
                f = -f;
            }
            f.to_bits().wrapping_add(io[l / wave * wave + 3])
        })
        .collect()
}

#[test]
fn emitted_kernel_passes_the_sdk_validator() {
    let c = sample_kernel(64);
    if let Err(e) = validate::validate(&c) {
        let dis = validate::disassemble(&c).unwrap_or_else(|d| d);
        panic!("validator rejected the emitted kernel:\n{e}\n--- disassembly ---\n{dis}");
    }
}

#[test]
fn emitted_kernel_disassembles_like_dxc_output() {
    let dis = validate::disassemble(&sample_kernel(64)).expect("disassembly");
    for needle in [
        "@dx.op.createHandleFromBinding(i32 217",
        "@dx.op.rawBufferLoad.i32(i32 139",
        "@dx.op.waveReadLaneAt.i32(i32 117",
        "phi float",
        "!{i32 64, i32 1, i32 1}",
    ] {
        assert!(dis.contains(needle), "missing `{needle}` in:\n{dis}");
    }
}

#[test]
fn emitted_kernel_runs_on_the_gpu() {
    let wave = match (runner::wave64_supported(), runner::wave32_supported()) {
        (Ok(true), _) => 64,
        (_, Ok(true)) => 32,
        (Err(e), _) => return runner::skip_test(&e),
        _ => return runner::skip_test("neither wave64 nor wave32 support"),
    };
    let c = sample_kernel(wave);
    validate::validate(&c).expect("valid DXIL");
    // io dwords 0..64 are the SGPR slots the runner uploads first.
    let mut sgpr = vec![0u32; 128];
    for (l, v) in sgpr.iter_mut().take(64).enumerate() {
        *v = match l % 4 {
            0 => l as u32,                     // small integers (≤ 5 stay positive)
            1 => (l as f32 * 0.5).to_bits(),   // floats, all > 5 as integers → negated
            2 => 3,
            _ => (-(l as f32)).to_bits(),
        };
    }
    let state = runner::KernelState { sgpr: sgpr.clone(), vgpr: vec![vec![0; 256]; 64], scc: 0 };
    let r = runner::run(&c, &state, &[]).expect("GPU run");
    assert_eq!(&r.raw[..64], &expected(&sgpr, wave as usize)[..]);
}

/// Blocks created out of dominance order (a block used as a branch target
/// before the block that dominates it exists), an unreachable block feeding a
/// phi, and `replace_all_uses` on a trivial phi: the writer lays blocks out in
/// reverse post-order, drops the unreachable one and its phi edge, and the
/// result validates and runs (res[lane] = (lane << 2) + 7 + io[lane]).
fn out_of_order_kernel() -> Vec<u8> {
    let mut s = Shader::new(Stage::Compute, (6, 6));
    let io = s.raw_uav(0, 0);
    let res = s.raw_uav(1, 0);
    let lane = s.thread_id_in_group(0);
    let two = s.m.i32c(2);
    let off = s.m.bin(BinOp::Shl, lane, two);
    let entry = s.m.current_block();
    // created first, reached last
    let tail = s.m.new_block();
    let dead = s.m.new_block();
    let mid = s.m.new_block();
    s.m.br(mid);

    s.m.set_block(mid);
    let seven = s.m.i32c(7);
    let v = s.m.bin(BinOp::Add, off, seven);
    let x = s.raw_load(io, off, 1)[0];
    s.m.br(tail);

    s.m.set_block(dead);
    let junk = s.m.i32c(0xdead);
    s.m.br(tail);

    s.m.set_block(tail);
    let i32 = s.m.i32();
    let trivial = s.m.phi(i32); // only ever `v`: replaced below
    s.m.add_incoming(trivial, v, mid);
    let p = s.m.phi(i32);
    s.m.add_incoming(p, x, mid);
    s.m.add_incoming(p, junk, dead);
    let sum = s.m.bin(BinOp::Add, trivial, p);
    s.raw_store(res, off, &[sum]);
    s.m.replace_all_uses(trivial, v);
    let _ = entry;
    s.finish([64, 1, 1], None)
}

#[test]
fn blocks_are_written_in_reverse_post_order() {
    let c = out_of_order_kernel();
    if let Err(e) = validate::validate(&c) {
        let dis = validate::disassemble(&c).unwrap_or_else(|d| d);
        panic!("validator rejected the out-of-order kernel:\n{e}\n--- disassembly ---\n{dis}");
    }
    let dis = validate::disassemble(&c).expect("disassembly");
    assert!(!dis.contains("57005"), "the unreachable block's constant must be gone:\n{dis}");
    match runner::wave32_supported().or_else(|_| runner::wave64_supported()) {
        Ok(true) => {}
        Ok(false) => return runner::skip_test("no wave32 / wave64"),
        Err(e) => return runner::skip_test(&e),
    }
    let mut sgpr = vec![0u32; 128];
    for (l, v) in sgpr.iter_mut().take(64).enumerate() {
        *v = 1000 + l as u32 * 3;
    }
    let state = runner::KernelState { sgpr: sgpr.clone(), vgpr: vec![vec![0; 256]; 64], scc: 0 };
    let r = runner::run(&c, &state, &[]).expect("GPU run");
    let want: Vec<u32> = (0..64).map(|l| ((l as u32) << 2) + 7 + sgpr[l]).collect();
    assert_eq!(&r.raw[..64], &want[..]);
}

#[test]
fn f16_default_rounding_is_round_to_nearest_even() {
    match runner::wave32_supported().or_else(|_| runner::wave64_supported()) {
        Ok(true) => {}
        Ok(false) => return runner::skip_test("no wave32 / wave64"),
        Err(e) => return runner::skip_test(&e),
    }
    const V: u64 = 256;
    let mut code = Vec::new();
    for (n, e, f) in [
        ("V_CVT_F16_F32", "ENC_VOP1", &[("VDST", 10), ("SRC0", V)][..]),
        ("V_MUL_F16", "ENC_VOP2", &[("VDST", 11), ("SRC0", V + 1), ("VSRC1", 2)][..]),
        ("V_PK_MUL_F16", "ENC_VOP3P", &[("VDST", 12), ("SRC0", V + 1), ("SRC1", V + 2), ("OP_SEL_HI", 3)][..]),
        ("S_ENDPGM", "ENC_SOPP", &[][..]),
    ] {
        code.extend(crate::gpu::rdna2::encode(n, e, f, None).unwrap());
    }
    let opt = super::kernel::KernelOptions { wave64: false, split: false, mem_windows: 1, sampler_base: 0 };
    let dxil = super::kernel::translate(&code, opt).expect("direct DXIL translate");

    let cases: [(f32, u16); 7] = [
        (f32::from_bits(3075.0f32.to_bits() - 1), 0x6A01),
        (3075.0, 0x6A02),
        (f32::from_bits(3075.0f32.to_bits() + 1), 0x6A02),
        (3081.0, 0x6A04),
        (-3075.0, 0xEA02),
        (65520.0, 0x7C00),
        (f32::from_bits(0x3300_0001), 0x0001),
    ];
    let mut st = runner::KernelState::default();
    st.sgpr[127] = 0;
    for (lane, &(value, _)) in cases.iter().enumerate() {
        st.vgpr[lane][0] = value.to_bits();
    }
    st.vgpr[0][1] = 0xEA00_6A00;
    st.vgpr[0][2] = 0x3C01_3C01;

    let r = runner::run(&dxil, &st, &[]).expect("GPU run");
    let actual: [u16; 7] = std::array::from_fn(|lane| r.vgpr[lane][10] as u16);
    let expected = cases.map(|(_, bits)| bits);
    let product = r.vgpr[0][11] as u16;
    let packed = r.vgpr[0][12];
    assert_eq!(
        (actual, product, packed),
        (expected, 0x6A02, 0xEA02_6A02),
        "conversion {actual:04x?}, multiplication {product:04x}, packed multiplication {packed:08x}",
    );
}
