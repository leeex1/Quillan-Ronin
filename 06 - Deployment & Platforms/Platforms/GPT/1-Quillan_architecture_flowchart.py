class ACEFlowchartNode:
    def __init__(self, id, label, category, attributes=None):
        self.id = id
        self.label = label
        self.category = category
        self.attributes = attributes or {}
        self.connections = []

    def connect(self, other_node):
        self.connections.append(other_node)


class ACEOperationalFlowchart:
    def __init__(self):
        self.nodes = {}

    def add_node(self, id, label, category, attributes=None):
        node = ACEFlowchartNode(id, label, category, attributes)
        self.nodes[id] = node
        return node

    def connect_nodes(self, from_id, to_id):
        if from_id in self.nodes and to_id in self.nodes:
            self.nodes[from_id].connect(self.nodes[to_id])

    def summary(self):
        for node_id, node in self.nodes.items():
            print(f"[{node.category}] {node.label} ({node.id})")
            for conn in node.connections:
                print(f"  -> {conn.label} ({conn.id})")


# Full Quillan Operational Flowchart
flowchart = ACEOperationalFlowchart()

# Input pipeline
flowchart.add_node("A", "INPUT RECEPTION", "input")
flowchart.add_node("AIP", "ADAPTIVE PROCESSOR", "input")
flowchart.add_node("QI", "PROCESSING GATEWAY", "input")
flowchart.connect_nodes("A", "AIP")
flowchart.connect_nodes("AIP", "QI")

# Vector branches
vectors = [
    ("NLP", "LANGUAGE VECTOR"),
    ("EV", "SENTIMENT VECTOR"),
    ("CV", "CONTEXT VECTOR"),
    ("IV", "INTENT VECTOR"),
    ("MV", "META-REASONING VECTOR"),
    ("SV", "ETHICAL VECTOR"),
    ("PV", "PRIORITY VECTOR"),
    ("DV", "DECISION VECTOR"),
    ("VV", "VALUE VECTOR")
]

for vid, label in vectors:
    flowchart.add_node(vid, label, "vector")
    flowchart.connect_nodes("QI", vid)

flowchart.add_node("ROUTER", "ATTENTION ROUTER", "router")
for vid, _ in vectors:
    flowchart.connect_nodes(vid, "ROUTER")

# Final stages
cog_stages = [
    ("REF", "REFLECT"),
    ("SYN", "SYNTHESIZE"),
    ("FOR", "FORMULATE"),
    ("ACT", "ACTIVATE"),
    ("EXP", "EXPLAIN"),
    ("VER", "VERIFY"),
    ("FIN", "FINALIZE"),
    ("DEL", "DELIVER")
]

for i in range(len(cog_stages)):
    cid, label = cog_stages[i]
    flowchart.add_node(cid, label, "cognitive")
    if i == 0:
        flowchart.connect_nodes("ROUTER", cid)
    else:
        prev_id = cog_stages[i - 1][0]
        flowchart.connect_nodes(prev_id, cid)

if __name__ == "__main__":
    flowchart.summary()
