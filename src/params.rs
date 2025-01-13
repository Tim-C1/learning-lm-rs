use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::{tensor, SafeTensors};
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        let get_tensor = |name: &str| {
            let safe_tensor = safetensor
                .tensor(name)
                .expect("tensor name does not exist!");
            let (prefix, data, suffix) = unsafe { safe_tensor.data().align_to::<f32>() };
            assert_eq!(prefix.len(), 0, "data should be properly aligned");
            assert_eq!(suffix.len(), 0, "data should be properly aligned");
            let tensor = Tensor::<f32>::new(data.to_vec(), &safe_tensor.shape().to_vec());
            tensor
        };

        let n_layers = config.num_hidden_layers;
        let embedding_table = get_tensor("lm_head.weight");
        let mut rms_att_w = Vec::with_capacity(n_layers);
        let mut wq = Vec::with_capacity(n_layers);
        let mut wk = Vec::with_capacity(n_layers);
        let mut wv = Vec::with_capacity(n_layers);
        let mut wo = Vec::with_capacity(n_layers);
        let mut rms_ffn_w = Vec::with_capacity(n_layers);
        let mut w_up = Vec::with_capacity(n_layers);
        let mut w_gate = Vec::with_capacity(n_layers);
        let mut w_down = Vec::with_capacity(n_layers);
        let rms_out_w = get_tensor("model.norm.weight");
        let lm_head = get_tensor("lm_head.weight");

        for i in 0..n_layers {
            rms_att_w.push(get_tensor(&format!(
                "model.layers.{i}.input_layernorm.weight"
            )));

            // Self-Attn
            wq.push(get_tensor(&format!(
                "model.layers.{i}.self_attn.q_proj.weight"
            )));
            wk.push(get_tensor(&format!(
                "model.layers.{i}.self_attn.k_proj.weight"
            )));
            wv.push(get_tensor(&format!(
                "model.layers.{i}.self_attn.v_proj.weight"
            )));
            wo.push(get_tensor(&format!(
                "model.layers.{i}.self_attn.o_proj.weight"
            )));

            // Attention 后、FFN 前的 RMSNorm
            rms_ffn_w.push(get_tensor(&format!(
                "model.layers.{i}.post_attention_layernorm.weight"
            )));

            // MLP (FFN)
            w_up.push(get_tensor(&format!("model.layers.{i}.mlp.up_proj.weight")));
            w_gate.push(get_tensor(&format!(
                "model.layers.{i}.mlp.gate_proj.weight"
            )));
            w_down.push(get_tensor(&format!(
                "model.layers.{i}.mlp.down_proj.weight"
            )));
        }
        LLamaParams {
            embedding_table,
            rms_att_w,
            wq,
            wk,
            wv,
            wo,
            rms_ffn_w,
            w_up,
            w_gate,
            w_down,
            rms_out_w,
            lm_head,
        }
    }
}
