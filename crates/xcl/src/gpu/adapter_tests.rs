use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter1, IDXGIFactory4, DXGI_ADAPTER_DESC1, DXGI_ADAPTER_FLAG_SOFTWARE};

use super::{is_software, select_indices, Selector};

#[test]
fn selector_parsing() {
    assert_eq!(Selector::parse(""), Selector::Auto);
    assert_eq!(Selector::parse(" Auto "), Selector::Auto);
    assert_eq!(Selector::parse("WARP"), Selector::Warp);
    assert_eq!(Selector::parse("0"), Selector::Index(0));
    assert_eq!(Selector::parse("1"), Selector::Index(1));
    assert_eq!(Selector::parse("007"), Selector::Name("007".into()));
    assert_eq!(Selector::parse("+1"), Selector::Name("+1".into()));
    assert_eq!(Selector::parse("Radeon"), Selector::Name("radeon".into()));
    assert_eq!(Selector::parse(" RTX 3080 "), Selector::Name("rtx 3080".into()));
}

#[test]
fn selector_matching() {
    let nv = "NVIDIA GeForce RTX 3080";
    let amd = "AMD Radeon(TM) Graphics";
    assert!(Selector::Auto.allows(0, nv) && Selector::Auto.allows(1, amd));
    assert!(!Selector::Warp.allows(0, nv));
    assert!(Selector::Index(1).allows(1, amd) && !Selector::Index(1).allows(0, nv));
    let radeon = Selector::parse("radeon");
    assert!(radeon.allows(1, amd) && !radeon.allows(0, nv));
    assert!(Selector::parse("geforce rtx").allows(0, nv));
}

#[test]
fn numeric_selector_falls_back_to_model_token_when_out_of_range() {
    let gpus = [
        "NVIDIA GeForce RTX 5070",
        "AMD Radeon™ RX 7900XTX",
        "Intel® Arc™ A770 Graphics",
    ];
    // In-range numbers still pick by index.
    assert_eq!(select_indices(&gpus, &Selector::parse("0")), vec![0]);
    assert_eq!(select_indices(&gpus, &Selector::parse("1")), vec![1]);
    assert_eq!(select_indices(&gpus, &Selector::parse("2")), vec![2]);

    // Out-of-range numbers match standalone numeric model tokens (bounded by non-digits).
    assert_eq!(select_indices(&gpus, &Selector::parse("5070")), vec![0]);
    assert_eq!(select_indices(&gpus, &Selector::parse("7900")), vec![1]);
    assert_eq!(select_indices(&gpus, &Selector::parse("770")), vec![2]);

    // Empty adapter list produces no match.
    let empty: [&str; 0] = [];
    assert!(select_indices(&empty, &Selector::parse("0")).is_empty());

    // Partial digit substrings inside a larger model number must NOT match.
    let two = ["NVIDIA GeForce RTX 2080", "AMD Radeon(TM) Graphics"];
    assert!(select_indices(&two, &Selector::parse("2")).is_empty());
    assert!(select_indices(&two, &Selector::parse("208")).is_empty());
    assert!(select_indices(&two, &Selector::parse("80")).is_empty());
    assert_eq!(select_indices(&two, &Selector::parse("2080")), vec![0]);
}

#[test]
fn microsoft_basic_render_driver_without_software_flag_is_still_software() {
    let flagged = DXGI_ADAPTER_DESC1 { Flags: DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32, ..Default::default() };
    assert!(is_software(&flagged));

    // Headless Windows Server VMs (e.g. windows-2025 CI) report Microsoft Basic Render Driver
    // with Flags == 0, VendorId 0x1414, DeviceId 0x008c.
    let unflagged_basic = DXGI_ADAPTER_DESC1 { VendorId: 0x1414, DeviceId: 0x008c, Flags: 0, ..Default::default() };
    assert!(is_software(&unflagged_basic));

    let nvidia = DXGI_ADAPTER_DESC1 { VendorId: 0x10de, DeviceId: 0x2c02, Flags: 0, ..Default::default() };
    let amd = DXGI_ADAPTER_DESC1 { VendorId: 0x1002, DeviceId: 0x744c, Flags: 0, ..Default::default() };
    let other_ms = DXGI_ADAPTER_DESC1 { VendorId: 0x1414, DeviceId: 0x0001, Flags: 0, ..Default::default() };
    assert!(!is_software(&nvidia));
    assert!(!is_software(&amd));
    assert!(!is_software(&other_ms));

    unsafe {
        if let Ok(factory) = CreateDXGIFactory1::<IDXGIFactory4>() {
            if let Ok(warp) = factory.EnumWarpAdapter::<IDXGIAdapter1>() {
                let desc = warp.GetDesc1().expect("WARP GetDesc1");
                assert!(is_software(&desc));
            }
        }
    }
}
