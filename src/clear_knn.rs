use crate::{BatcherSort, ClearCmp, ClearItem};
use rand::prelude::SliceRandom;
use std::collections::HashMap;

const BEST_MODEL_TRIES: usize = 10000;

// Repeatedly train and find the set that has the highest accuracy
// the accuracy is computed for all possible test vectors
pub fn find_best_model(
    model_size: usize,
    output_test_size: usize,
    k: usize,
    rows: &Vec<Vec<u64>>,
) -> (Vec<Vec<u64>>, Vec<u64>, Vec<Vec<u64>>, Vec<u64>, f64) {
    let mut final_model_vec: Vec<Vec<u64>> = vec![];
    let mut final_test_vec: Vec<Vec<u64>> = vec![];
    let mut final_model_labels: Vec<u64> = vec![];
    let mut final_test_labels: Vec<u64> = vec![];
    let mut highest_accuracy: usize = 0;

    let mut rng = rand::thread_rng();
    let test_size = rows.len() - model_size;

    for _ in 0..BEST_MODEL_TRIES {
        // shuffle and split model/test vector
        let mut rows = rows.clone();
        rows.shuffle(&mut rng);
        let (model_vec, model_labels, test_vec, test_labels) =
            split_model_test(model_size, test_size, rows);

        // do knn and check accuracy
        let mut oks: usize = 0;
        for (target, expected) in test_vec.iter().zip(&test_labels) {
            let (out, _) = run_knn(k, &model_vec, &model_labels, &target);
            let out_labels: Vec<_> = out.iter().map(|l| l.class).collect();
            let res = majority(&out_labels);
            if res == *expected {
                oks += 1;
            }
        }

        // check if our accuracy is higher
        if oks > highest_accuracy {
            final_model_vec = model_vec;
            final_model_labels = model_labels;
            final_test_vec = test_vec[..output_test_size].to_vec();
            final_test_labels = test_labels[..output_test_size].to_vec();
            highest_accuracy = oks;
        }
    }

    (
        final_model_vec,
        final_model_labels,
        final_test_vec,
        final_test_labels,
        highest_accuracy as f64 / test_size as f64,
    )
}

/// Split the feature vectors into a training and testing set.
/// The feature vectors are specified in `rows` and the last
/// element of every vector is the label.
pub fn split_model_test(
    model_size: usize,
    test_size: usize,
    rows: Vec<Vec<u64>>,
) -> (Vec<Vec<u64>>, Vec<u64>, Vec<Vec<u64>>, Vec<u64>) {
    let mut model_vec: Vec<Vec<u64>> = vec![];
    let mut test_vec: Vec<Vec<u64>> = vec![];
    let mut model_labels: Vec<u64> = vec![];
    let mut test_labels: Vec<u64> = vec![];

    for (i, mut row) in rows.into_iter().enumerate() {
        let last = row.pop().unwrap();
        if i < model_size {
            model_vec.push(row);
            model_labels.push(last);
        } else if i >= model_size && i < model_size + test_size {
            test_vec.push(row);
            test_labels.push(last);
        } else {
            break;
        }
    }

    (model_vec, model_labels, test_vec, test_labels)
}

fn squared_distance(xs: &[u64], ys: &[u64]) -> u64 {
    xs.iter()
        .zip(ys)
        .map(|(x, y)| {
            let diff = if x > y { x - y } else { y - x };
            diff * diff
        })
        .sum()
}

pub fn distances(data: &[Vec<u64>], target: &[u64]) -> Vec<u64> {
    data.iter().map(|x| squared_distance(x, target)).collect()
}

/// Run the k-NN classification algorithm.
pub fn run_knn(
    k: usize,
    model_vec: &[Vec<u64>],
    labels: &[u64],
    target: &[u64],
) -> (Vec<ClearItem>, u64) {
    let distances: Vec<_> = distances(model_vec, target)
        .iter()
        .zip(labels.iter())
        .map(|(value, class)| ClearItem {
            value: *value,
            class: *class,
        })
        .collect();
    let max_dist = distances.iter().map(|item| item.value).max().unwrap();
    let mut batcher = BatcherSort::new_k(ClearCmp::boxed(distances), k);
    batcher.sort();
    (batcher.inner()[0..k].to_vec(), max_dist)
}

pub fn majority(vs: &[u64]) -> u64 {
    assert!(vs.len() > 0);
    let max = vs
        .iter()
        .fold(HashMap::<u64, usize>::new(), |mut m, x| {
            *m.entry(*x).or_default() += 1;
            m
        })
        .into_iter()
        .max_by_key(|(_, v)| *v)
        .map(|(k, _)| k);
    max.unwrap()
}
