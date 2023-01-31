use ppknn::*;

fn main() {
    for e in 0..10 {
        let k = 1 << e;
        let n = 1 << 10;
        let mut batcher = BatcherSort::new_k(vec![0; n], k);
        batcher.sort();
        println!("{}", batcher.comparisons());
    }
}
