import json
from typing import List, Dict, Optional

class FlowNode:
    def __init__(self, node_id: str, name: str, description: List[str], parent: Optional[str], children: List[str], node_class: str):
        self.node_id = node_id
        self.name = name
        self.description = description
        self.parent = parent
        self.children = children
        self.node_class = node_class

    def __repr__(self):
        return f"FlowNode({self.node_id}, {self.name}, class={self.node_class})"

class ACEFlowchart:
    def __init__(self):
        self.nodes: Dict[str, FlowNode] = {}

    def add_node(self, node_id: str, name: str, description: List[str], parent: Optional[str], children: List[str], node_class: str):
        self.nodes[node_id] = FlowNode(node_id, name, description, parent, children, node_class)

    def get_node(self, node_id: str) -> Optional[FlowNode]:
        return self.nodes.get(node_id)

    def display_flow(self):
        for node_id, node in self.nodes.items():
            print(f"{node_id}: {node.name} -> Children: {node.children}")

    def find_path_to_root(self, node_id: str) -> List[str]:
        path = []
        current = self.get_node(node_id)
        while current:
            path.insert(0, current.name)
            current = self.get_node(current.parent) if isinstance(current.parent, str) else None
        return path

    def build_from_mermaid(self, mermaid_lines: List[str]):
        for line in mermaid_lines:
            if "-->" in line:
                src, tgt = [x.strip() for x in line.split("-->")]
                src_id = src.split("[")[0].strip()
                tgt_id = tgt.split("[")[0].strip()
                if src_id not in self.nodes:
                    self.nodes[src_id] = FlowNode(src_id, src_id, [], None, [], "unknown")
                if tgt_id not in self.nodes:
                    self.nodes[tgt_id] = FlowNode(tgt_id, tgt_id, [], src_id, [], "unknown")
                self.nodes[src_id].children.append(tgt_id)
                self.nodes[tgt_id].parent = src_id

# Example usage
if __name__ == "__main__":
    mermaid_example = [
        "A[Input Reception] --> AIP[Adaptive Processor]",
        "AIP --> QI[Processing Gateway]",
        "QI --> NLP[Language Vector]",
        "QI --> EV[Sentiment Vector]",
        "NLP --> ROUTER[Attention Router]",
        "EV --> ROUTER"
    ]
    ace_flow = ACEFlowchart()
    ace_flow.build_from_mermaid(mermaid_example)
    ace_flow.display_flow()
    print("\nPath to root for 'ROUTER':", " -> ".join(ace_flow.find_path_to_root("ROUTER")))
