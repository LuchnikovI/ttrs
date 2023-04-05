use linwrap::{
    NDArray,
    ParPtrWrapper,
};
use rawpointer::PointerExt;
//use rayon::prelude::{IntoParallelIterator, ParallelIterator, IndexedParallelIterator};

static XGRIDSIZE: usize = 70000;
static YGRIDSIZE: usize = 100000;
static XSTART: f64 = -0.5;
static YSTART: f64 = -1.;
static XEND: f64 = 0.5;
static YEND: f64 = 1.;
static RANK: usize = 3;


#[inline]
fn fn_to_integrate(x: f64, y: f64) -> f64 {
    (26. * x).cos() * (20. * y).cos() + (x - 0.42).powi(2) * (y - 0.59).powi(2) + 1.
}

#[inline]
unsafe fn get_cols(cols_nums: &[usize]) -> Vec<f64>
{
    let dx = (XEND - XSTART) / XGRIDSIZE as f64;
    let dy = (YEND - YSTART) / YGRIDSIZE as f64;
    let mut buff = Vec::with_capacity(cols_nums.len() * YGRIDSIZE);
    buff.set_len(cols_nums.len() * YGRIDSIZE);
    let ptr = ParPtrWrapper(buff.as_mut_ptr());
    (0..RANK).into_iter().zip(cols_nums.into_iter()).for_each(|(i, o)| {
        for j in 0..YGRIDSIZE {
            *ptr.add(i + j * RANK).0 = fn_to_integrate(
                XSTART + dx / 2. + dx * *o as f64,
                YSTART + dy / 2. + dy *  j as f64,
            );
        }
    });
    buff
}

#[inline]
unsafe fn get_rows(rows_nums: &[usize]) -> Vec<f64>
{
    let dx = (XEND - XSTART) / XGRIDSIZE as f64;
    let dy = (YEND - YSTART) / YGRIDSIZE as f64;
    let mut buff = Vec::with_capacity(rows_nums.len() * XGRIDSIZE);
    buff.set_len(rows_nums.len() * XGRIDSIZE);
    let ptr = ParPtrWrapper(buff.as_mut_ptr());
    (0..RANK).into_iter().zip(rows_nums.into_iter()).for_each(|(i, o)| {
        for j in 0..XGRIDSIZE {
            *ptr.add(i + j * RANK).0 = fn_to_integrate(
                XSTART + dx / 2. + dx *  j as f64,
                YSTART + dy / 2. + dy * *o as f64,
            );
        }
    });
    buff
}

#[inline]
unsafe fn get_intersec(cols_nums: &[usize], rows_nums: &[usize]) -> Vec<f64>
{
    let dx = (XEND - XSTART) / XGRIDSIZE as f64;
    let dy = (YEND - YSTART) / YGRIDSIZE as f64;
    let mut buff = Vec::with_capacity(rows_nums.len() * cols_nums.len());
    buff.set_len(rows_nums.len() * cols_nums.len());
    let ptr = ParPtrWrapper(buff.as_mut_ptr());
    (0..RANK).into_iter().zip(rows_nums.into_iter()).for_each(|(i, o1)| {
        (0..RANK).into_iter().zip(cols_nums.into_iter()).for_each(|(j, o2)| {
            *ptr.add(i * RANK + j).0 = fn_to_integrate(
                XSTART + dx / 2. + dx * *o2 as f64,
                YSTART + dy / 2. + dy * *o1 as f64,
            );
        })
    });
    buff
}

fn main() {
    let mut rows_order: Vec<usize> = (0..RANK).map(|i| { i * 100 }).collect();
    let mut cols_order = rows_order.clone();
    let mut counter = 0;
    unsafe
    {
        loop
        {
            let mut cols_buff = get_cols(&cols_order);
            counter += YGRIDSIZE * RANK;
            let cols = NDArray::from_mut_slice(&mut cols_buff, [RANK, YGRIDSIZE]).unwrap();
            let (mut cols_tr_buff, _) = cols.transpose([1, 0]).unwrap().gen_f_array();
            let cols_tr = NDArray::from_mut_slice(&mut cols_tr_buff, [YGRIDSIZE, RANK]).unwrap();
            let mut new_rows_order = cols_tr.maxvol(0., None, None).unwrap();
            new_rows_order.resize(RANK, 0);
            if new_rows_order == rows_order {
                break;
            } else {
                rows_order = new_rows_order;
            }
            let mut rows_buff = get_rows(&rows_order);
            counter += XGRIDSIZE * RANK;
            let rows = NDArray::from_mut_slice(&mut rows_buff, [RANK, XGRIDSIZE]).unwrap();
            let (mut rows_tr_buff, _) = rows.transpose([1, 0]).unwrap().gen_f_array();
            let rows_tr = NDArray::from_mut_slice(&mut rows_tr_buff, [XGRIDSIZE, RANK]).unwrap();
            let mut new_cols_order = rows_tr.maxvol(0., None, None).unwrap();
            new_cols_order.resize(RANK, 0);
            if new_cols_order == cols_order {
                break;
            } else {
                cols_order = new_cols_order;
            }
        }
        let mut cols_buff = get_cols(&cols_order);
        let cols = NDArray::from_mut_slice(&mut cols_buff, [RANK, YGRIDSIZE]).unwrap();
        let rows_buff = get_rows(&rows_order);
        let rows = NDArray::from_slice(&rows_buff, [RANK, XGRIDSIZE]).unwrap();
        let mut intersec_buff = get_intersec(&cols_order, &rows_order);
        let intersec = NDArray::from_mut_slice(&mut intersec_buff, [RANK, RANK]).unwrap();
        intersec.solve(cols).unwrap();
        let buff_lhs_ones = vec![1f64; YGRIDSIZE];
        let lhs_ones = NDArray::from_slice(&buff_lhs_ones, [1, YGRIDSIZE]).unwrap();
        let mut lhs_buff = vec![0f64; RANK];
        let lhs = NDArray::from_mut_slice(&mut lhs_buff, [1, RANK]).unwrap();
        lhs.matmul_inplace(lhs_ones, cols.transpose([1, 0]).unwrap()).unwrap();
        let buff_rhs_ones = vec![1f64; XGRIDSIZE];
        let rhs_ones = NDArray::from_slice(&buff_rhs_ones, [1, XGRIDSIZE]).unwrap();
        let mut rhs_buff = vec![0f64; RANK];
        let rhs = NDArray::from_mut_slice(&mut rhs_buff, [RANK, 1]).unwrap();
        rhs.matmul_inplace(rows, rhs_ones.transpose([1, 0]).unwrap()).unwrap();
        let mut result_buff = vec![0f64; 1];
        let result = NDArray::from_mut_slice(&mut result_buff, [1, 1]).unwrap();
        result.matmul_inplace(lhs, rhs).unwrap();
        let dx = (XEND - XSTART) / XGRIDSIZE as f64;
        let dy = (YEND - YSTART) / YGRIDSIZE as f64;
        println!("Number of the target function calls: {:?}", counter);
        println!("Naive number of function calls: {:?}", XGRIDSIZE * YGRIDSIZE);
        println!("Cross approximation based integration value: {:?},", *result.at([0, 0]).unwrap() * dx * dy);
        println!("Wolfram alpha result: {:?}.", 2.35693);
    }
}