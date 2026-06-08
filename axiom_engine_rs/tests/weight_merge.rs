use std::path::PathBuf;

use axiom_engine::weight_merge::{
    merge_checkpoint_files, merge_layer_stacks, write_test_cache, LayerWeights,
};

fn temp_file(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "axiom_weight_merge_{name}_{}_{}.bin",
        std::process::id(),
        uuid::Uuid::new_v4()
    ))
}

#[test]
fn task_vector_merge_preserves_tensor_shapes() {
    let a = vec![
        LayerWeights {
            shape: vec![2, 2],
            data: vec![1.2, 0.0, 0.0, 1.2],
        },
        LayerWeights {
            shape: vec![2, 2],
            data: vec![0.8, 0.0, 0.0, 0.8],
        },
    ];
    let b = vec![
        LayerWeights {
            shape: vec![2, 2],
            data: vec![1.4, 0.0, 0.0, 1.4],
        },
        LayerWeights {
            shape: vec![2, 2],
            data: vec![1.0, 0.0, 0.0, 1.0],
        },
    ];

    let merged = merge_layer_stacks(&[a, b], 0.5).unwrap();

    assert_eq!(merged.len(), 2);
    assert_eq!(merged[0].shape, vec![2, 2]);
    assert_eq!(merged[1].shape, vec![2, 2]);
    assert_eq!(merged[0].data.len(), 4);
    assert_eq!(merged[1].data.len(), 4);
}

#[test]
fn merge_checkpoint_files_writes_merged_cache() {
    let a_path = temp_file("a");
    let b_path = temp_file("b");
    let out_path = temp_file("out");
    write_test_cache(
        &a_path,
        "a",
        vec![LayerWeights {
            shape: vec![2, 2],
            data: vec![1.2, 0.0, 0.0, 1.2],
        }],
    );
    write_test_cache(
        &b_path,
        "b",
        vec![LayerWeights {
            shape: vec![2, 2],
            data: vec![1.4, 0.0, 0.0, 1.4],
        }],
    );

    let summary =
        merge_checkpoint_files(&[a_path.clone(), b_path.clone()], &out_path, 0.5).unwrap();

    assert_eq!(summary.input_files, 2);
    assert_eq!(summary.input_sessions, 2);
    assert_eq!(summary.layers, 1);
    assert_eq!(summary.d_model, 2);
    assert!(out_path.exists());

    let _ = std::fs::remove_file(a_path);
    let _ = std::fs::remove_file(b_path);
    let _ = std::fs::remove_file(out_path);
}
