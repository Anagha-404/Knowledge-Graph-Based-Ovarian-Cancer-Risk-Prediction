from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain.config import ModelConfig


def explain_model(model, data, node_idx=0):

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=ModelConfig(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        ),
    )

    explanation = explainer(
        data.x,
        data.edge_index,
        index=node_idx
    )

    print("\nEXPLANATION:")

    # 🔥 FILTER IMPORTANT EDGES ONLY
    # get top important edges
    edge_mask = explanation.edge_mask

    # get top 10 important edges
    topk = 10
    top_indices = edge_mask.topk(topk).indices

    print("\nTop important edge scores:")
    print(edge_mask[top_indices])