fn main()
{
    let dst = cmake::Config::new("arpack-ng")
        .define("ICB", "OFF")
        .define("CMAKE_Fortran_FLAGS", "-fPIC")
        .build();
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=dylib=arpack");
}