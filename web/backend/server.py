from flask import Flask, jsonify, request
from flask_cors import CORS
import torch
import numpy as np
import json
from torch_geometric.data import Batch

# Import local modules
from main_model import RNA_ClassQuery_Model
from human import run_linearfold, build_edge_index_from_structure, MOD_NAMES


def one_hot_encode_sequence(sequence: str) -> np.ndarray:
    """
    Convert RNA sequence string to one-hot encoding.

    Args:
        sequence: RNA sequence string (A, C, G, U)

    Returns:
        One-hot encoded array of shape (len(sequence), 4)
    """
    one_hot_mapping = {
        'A': [1., 0., 0., 0.],
        'C': [0., 1., 0., 0.],
        'G': [0., 0., 1., 0.],
        'U': [0., 0., 0., 1.],
        'T': [0., 0., 0., 1.],  # Treat T as U
        'N': [0., 0., 0., 0.]
    }

    one_hot = np.zeros((len(sequence), 4), dtype=np.float32)
    for i, nucleotide in enumerate(sequence.upper()):
        if nucleotide in one_hot_mapping:
            one_hot[i] = one_hot_mapping[nucleotide]
        else:
            one_hot[i] = [0., 0., 0., 0.]  # Unknown nucleotide

    return one_hot

# 初始化 Flask app
app = Flask(__name__)
# 设置 CORS，允许来自前端（例如 http://localhost:5173）的跨域请求
CORS(app)

# ============================================================================
# Global Model Loading (Load once at startup)
# ============================================================================

print("Loading model and configuration...")

# Load configuration
with open('json/human.json', 'r') as f:
    config = json.load(f)

model_cfg = config['model']

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = RNA_ClassQuery_Model(
    cnn_hidden_dim=model_cfg['cnn_hidden_dim'],
    cnn_kernel_sizes=tuple(model_cfg['cnn_kernel_sizes']),
    cnn_dropout=model_cfg['cnn_dropout'],
    gcn_hidden_dim=model_cfg['gcn_hidden_dim'],
    gcn_out_channels=model_cfg['gcn_out_channels'],
    gcn_num_layers=model_cfg['gcn_num_layers'],
    gcn_dropout=model_cfg['gcn_dropout'],
    num_classes=model_cfg['num_classes'],
    num_attn_heads=model_cfg['num_attn_heads'],
    attn_dropout=model_cfg['attn_dropout'],
    use_simple_pooling=model_cfg['use_simple_pooling'],
    use_hierarchical=model_cfg['use_hierarchical'],
    use_layer_norm=model_cfg['use_layer_norm']
)

# Load model weights
checkpoint_path = '/home/dc/vscode/vscode20260119/rgcnformer_sum/web/backend/epoch_040.pt'
checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(f"Model loaded successfully from {checkpoint_path}")
print("Model is ready for predictions!")

# ============================================================================
# Health Check Endpoint
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "model_loaded": True,
        "device": str(device),
        "checkpoint": checkpoint_path
    })

# ============================================================================
# Prediction Endpoint
# ============================================================================

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    接收来自前端的RNA序列，调用模型进行预测，返回预测结果。
    """
    # 从请求的JSON body中获取数据
    data = request.get_json()
    original_sequence = ""
    if data:
        original_sequence = data.get('sequence', '')
        print(f"Received sequence from frontend: {original_sequence[:50]}... (length: {len(original_sequence)})")

    # Validate sequence
    if not original_sequence:
        return jsonify({"error": "No sequence provided"}), 400

    # Store original sequence for response
    sequence = original_sequence

    # For shorter sequences, pad to 1001; for longer sequences, truncate
    # The model expects exactly 1001 nucleotides
    TARGET_LENGTH = 1001
    seq_len = len(sequence)

    # Track padding/trimming for index remapping
    left_padding = 0
    left_trimming = 0

    if seq_len != TARGET_LENGTH:
        print(f"Warning: Sequence length is {seq_len}, expected {TARGET_LENGTH}")
        if seq_len < TARGET_LENGTH:
            # Pad with 'N' (unknown nucleotide) at the center
            padding_needed = TARGET_LENGTH - seq_len
            left_pad = padding_needed // 2
            right_pad = padding_needed - left_pad
            left_padding = left_pad
            sequence = 'N' * left_pad + sequence + 'N' * right_pad
            print(f"Padded sequence to {TARGET_LENGTH} with {padding_needed} 'N's (left: {left_pad}, right: {right_pad})")
        else:
            # Truncate from both sides (center the sequence)
            excess = seq_len - TARGET_LENGTH
            left_trim = excess // 2
            right_trim = excess - left_trim
            left_trimming = left_trim
            sequence = sequence[left_trim:seq_len - right_trim]
            print(f"Trimmed sequence from {seq_len} to {TARGET_LENGTH} (left: {left_trim}, right: {right_trim})")

    try:
        # Step 1: Call LinearFold to get secondary structure
        structures = run_linearfold([sequence])
        structure = structures[0]

        # Step 2: Build edge index from structure
        edge_index = build_edge_index_from_structure(sequence, structure)

        # Step 3: Prepare model input
        # Convert sequence to one-hot encoding
        x = one_hot_encode_sequence(sequence)
        x = torch.FloatTensor(x)  # Shape: [1001, 4]

        # Create batch tensor (single sample)
        batch = torch.zeros(len(sequence), dtype=torch.long)

        # Step 4: Run model inference
        with torch.no_grad():
            # Create Batch object
            data_batch = Batch(x=x, edge_index=edge_index, batch=batch)

            # Move to device
            data_batch = data_batch.to(device)

            # Get predictions
            if model_cfg['use_hierarchical']:
                logits_12class, logits_4class, attn_weights = model(
                    data_batch.x,
                    data_batch.edge_index,
                    data_batch.batch,
                    return_attention=True
                )
            else:
                logits_12class = model(data_batch)
                attn_weights = None

        # Step 5: Process and format output
        # Move to CPU and convert to numpy
        logits_12class = logits_12class.cpu().numpy()[0]  # [12]
        probs_12class = 1 / (1 + np.exp(-logits_12class))  # Sigmoid

        if model_cfg['use_hierarchical']:
            logits_4class = logits_4class.cpu().numpy()[0]  # [4]
            probs_4class = 1 / (1 + np.exp(-logits_4class))  # Sigmoid
            attn_weights = attn_weights.cpu().numpy()[0]  # [12, 1001]

        # Build classification tree
        group_names = ['A', 'C', 'G', 'U']
        group_to_classes = {
            'A': [0, 1, 7, 9, 10],
            'C': [2, 6, 8],
            'G': [3, 11],
            'U': [4, 5]
        }

        # Reverse mapping: class index to group name
        class_to_group = {}
        for group_name, class_indices in group_to_classes.items():
            for class_idx in class_indices:
                class_to_group[class_idx] = group_name

        # Optimal thresholds from training (Table 3 and 4)
        # 12-class thresholds
        thresholds_12class = {
            0: 0.510,   # Am
            1: 0.400,   # Atol
            2: 0.690,   # Cm
            3: 0.710,   # Gm
            4: 0.350,   # Tm
            5: 0.150,   # Y
            6: 0.120,   # ac4C
            7: 0.380,   # m1A
            8: 0.350,   # m5C
            9: 0.260,   # m6A
            10: 0.570,  # m6Am
            11: 0.130   # m7G
        }

        # 4-class thresholds (group-level)
        thresholds_4class = {
            0: 0.980,   # A
            1: 0.270,   # C
            2: 0.050,   # G
            3: 0.050    # U
        }

        # Step 5.1: Get initial predictions for 12-class and 4-class
        predictions_12class = {}
        for class_idx in range(12):
            class_prob = probs_12class[class_idx]
            class_threshold = thresholds_12class[class_idx]
            predictions_12class[class_idx] = bool(class_prob > class_threshold)

        predictions_4class = {}
        if model_cfg['use_hierarchical']:
            for group_idx in range(4):
                group_prob = probs_4class[group_idx]
                group_threshold = thresholds_4class[group_idx]
                predictions_4class[group_idx] = bool(group_prob > group_threshold)
        else:
            # If not hierarchical, all groups are False by default
            for group_idx in range(4):
                predictions_4class[group_idx] = False

        # Step 5.2: Apply hierarchical pruning (bottom-up)
        # Rule 1: If all children of a group are False, set the group to False
        for group_idx, group_name in enumerate(group_names):
            child_indices = group_to_classes[group_name]
            # Check if any child is predicted True
            has_any_child = any(predictions_12class[class_idx] for class_idx in child_indices)
            if not has_any_child:
                predictions_4class[group_idx] = False

        # Step 5.3: Apply hierarchical pruning (top-down)
        # Rule 2: If a group is False, set all its children to False
        for group_idx, group_name in enumerate(group_names):
            if not predictions_4class[group_idx]:
                child_indices = group_to_classes[group_name]
                for class_idx in child_indices:
                    predictions_12class[class_idx] = False

        # Step 5.4: Build classification tree with pruned predictions
        classification = {
            "name": "RNA Sequence",
            "isPredicted": True,
            "children": []
        }

        for group_idx, group_name in enumerate(group_names):
            group_predicted = predictions_4class[group_idx]

            children = []
            for class_idx in group_to_classes[group_name]:
                class_predicted = predictions_12class[class_idx]
                if class_predicted:
                    children.append({
                        "name": MOD_NAMES.get(class_idx, f"Class{class_idx}"),
                        "isPredicted": True
                    })

            classification["children"].append({
                "name": f"Group {group_name}",
                "isPredicted": group_predicted,
                "children": children
            })

        # Step 5.5: Identify active nucleotide groups based on pruned predictions
        # Find which nucleotide groups have at least one predicted modification
        active_groups = set()
        for group_idx, is_predicted in predictions_4class.items():
            if is_predicted:
                active_groups.add(group_names[group_idx])
        
        print(f"Active groups after pruning: {active_groups}")

        # Build attention data
        attention_data = {
            "sequence": original_sequence,  # Return original sequence
            "weights": []
        }

        if attn_weights is not None and active_groups:
            # Step 5.6: Calculate combined attention for predicted classes
            predicted_class_indices = [idx for idx, pred in predictions_12class.items() if pred]
            
            combined_attention = np.zeros(len(sequence))
            if predicted_class_indices:
                for class_idx in predicted_class_indices:
                    combined_attention += attn_weights[class_idx]
                combined_attention /= len(predicted_class_indices)

            # Step 5.7: Per-group Top-K selection
            K = 3  # Number of top sites to select from each active group
            all_top_sites = []

            for group in active_groups:
                # Create a mask specifically for the current group's nucleotides
                group_mask = np.array([1.0 if nucleotide == group else 0.0 for nucleotide in sequence.upper()])
                
                # Apply the group mask to the attention weights
                group_attention = combined_attention * group_mask
                
                # Set non-group positions to a very low value so they are not selected
                group_attention[group_mask == 0] = -np.inf
                
                # Determine how many sites to get for this group (min of K and available sites)
                num_available_sites = int(group_mask.sum())
                top_k_for_group = min(K, num_available_sites)

                if top_k_for_group > 0:
                    # Get the indices of the top-k highest scores for this group
                    top_indices_for_group = np.argsort(group_attention)[-top_k_for_group:][::-1]
                    
                    for pos in top_indices_for_group:
                        pos_int = int(pos)
                        score_float = float(combined_attention[pos]) # Use original combined attention for score
                        original_index = pos_int - left_padding + left_trimming

                        # Ensure the site is within the original sequence bounds
                        if 0 <= original_index < len(original_sequence):
                             all_top_sites.append({
                                "index": original_index,
                                "type": group, # Label with the group name
                                "score": score_float
                            })
            
            # Sort all collected sites by score in descending order
            all_top_sites.sort(key=lambda x: x["score"], reverse=True)
            attention_data["weights"] = all_top_sites

        # Build GCN graph data
        # Limit to original sequence for visualization
        edge_index_np = edge_index.cpu().numpy()
        edges = []
        max_edges = min(100, int(edge_index_np.shape[1]))  # Limit to 100 edges

        # Calculate the valid range in model coordinates
        valid_start = left_padding
        valid_end = left_padding + len(original_sequence)

        for i in range(max_edges):
            source = int(edge_index_np[0, i])
            target = int(edge_index_np[1, i])
            # Map model indices to original indices
            orig_source = source - left_padding + left_trimming
            orig_target = target - left_padding + left_trimming

            # Only include edges within original sequence bounds
            if (source < target and
                0 <= orig_source < len(original_sequence) and
                0 <= orig_target < len(original_sequence)):
                nuc_source = original_sequence[orig_source]
                nuc_target = original_sequence[orig_target]
                edges.append({
                    "source": f"{nuc_source}{orig_source}",
                    "target": f"{nuc_target}{orig_target}"
                })

        # Create nodes (limit to 50 or original sequence length)
        nodes = []
        max_nodes = min(50, len(original_sequence))
        for i in range(max_nodes):
            nuc = original_sequence[i]
            nodes.append({
                "id": f"{nuc}{i}",
                "label": f"位置{i}: {nuc}",
                "data": {"index": i, "type": nuc, "name": f"{'腺嘌呤' if nuc == 'A' else '胞嘧啶' if nuc == 'C' else '鸟嘌呤' if nuc == 'G' else '尿嘧啶'}"}
            })

        # Create a set of valid node IDs for filtering edges
        valid_node_ids = {node["id"] for node in nodes}

        # Filter edges to only include those that reference valid nodes
        valid_edges = [
            edge for edge in edges
            if edge["source"] in valid_node_ids and edge["target"] in valid_node_ids
        ]

        gcn_data = {
            "nodes": nodes,
            "edges": valid_edges
        }

        # Final response
        response = {
            "classification": classification,
            "attention": attention_data,
            "gcn": gcn_data
        }

        return jsonify(response)

    except Exception as e:
        import traceback
        error_msg = f"Prediction error: {str(e)}"
        print(f"ERROR: {error_msg}")
        print(f"Traceback:\n{traceback.format_exc()}")
        return jsonify({
            "error": error_msg,
            "detail": str(e),
            "type": type(e).__name__
        }), 500

if __name__ == '__main__':
    # 运行服务器在 5000 端口
    app.run(debug=True, port=5000)
