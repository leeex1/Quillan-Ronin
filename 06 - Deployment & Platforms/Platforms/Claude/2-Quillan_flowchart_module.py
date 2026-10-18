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
    def __init__(self, json_data: Dict):
        self.nodes = {}
        for node in json_data.get("nodes", []):
            node_id = node.get("NodeID")
            self.nodes[node_id] = FlowNode(
                node_id=node_id,
                name=node.get("NodeName"),
                description=node.get("Description", []),
                parent=node.get("ParentNode"),
                children=node.get("ChildNodes", []),
                node_class=node.get("Class")
            )

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

# Example usage
if __name__ == "__main__":
    with open("2-ace_flowchart.json") as f:
        data = json.load(f)
    ace_flow = ACEFlowchart(data)
    ace_flow.display_flow()
    print("\nPath to root for 'C1R1':", " -> ".join(ace_flow.find_path_to_root("C1R1")))
