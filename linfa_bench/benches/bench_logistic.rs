use linfa_datasets::iris;               // ← now available
use linfa::traits::{Fit, Predict};
use linfa_logistic::LogisticRegression;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn bench_training(c: &mut Criterion) {
    let dataset = iris();               // 150 samples, 4 features :contentReference[oaicite:4]{index=4}
    let (train, valid) = dataset.split_with_ratio(0.7);
    c.bench_function("rust_train", |b| {
        b.iter(|| {
            let model = LogisticRegression::default()
                .fit(black_box(&train))
                .unwrap();
            black_box(model);
        })
    });
}

pub fn bench_predict(c: &mut Criterion) {
    let dataset = iris();
    let (train, valid) = dataset.split_with_ratio(0.7);
    let model = LogisticRegression::default().fit(&train).unwrap();
    c.bench_function("rust_predict", |b| {
        b.iter(|| {
            let preds = model.predict(black_box(&valid));
            black_box(preds);
        })
    });
}

criterion_group!(benches, bench_training, bench_predict);
criterion_main!(benches);
