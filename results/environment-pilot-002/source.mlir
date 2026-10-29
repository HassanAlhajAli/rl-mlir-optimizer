module {
  func.func @kernel(%a: memref<32x32xf64>, %b: memref<32x32xf64>,
                    %c: memref<32x32xf64>) {
    %zero = arith.constant 0.0 : f64
    linalg.fill ins(%zero : f64) outs(%c : memref<32x32xf64>)
    linalg.generic {indexing_maps = [
      affine_map<(i,j,k)->(i,k)>, affine_map<(i,j,k)->(k,j)>,
      affine_map<(i,j,k)->(i,j)>],
      iterator_types = ["parallel", "parallel", "reduction"]}
      ins(%a, %b : memref<32x32xf64>, memref<32x32xf64>)
      outs(%c : memref<32x32xf64>) {
      ^bb0(%x: f64, %y: f64, %sum: f64):
        %p = arith.mulf %x, %y : f64
        %s = arith.addf %sum, %p : f64
        linalg.yield %s : f64
    }
    return
  }
}