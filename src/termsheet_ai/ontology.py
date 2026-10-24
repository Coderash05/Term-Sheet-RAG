import networkx as nx


def build_ontology_graph():
    G = nx.DiGraph()

    # Define Source Documents
    G.add_node("NVCA_Model_Term_Sheet", type="SourceDoc")
    G.add_node("YC_SAFE", type="SourceDoc")

    # Define Clause Families
    G.add_node("LiquidationPreference", type="ClauseFamily")
    G.add_node("BoardOfDirectors", type="ClauseFamily")
    G.add_node("Vesting", type="ClauseFamily")

    # Define Attributes
    G.add_node("Multiple", type="Attribute")
    G.add_node("Participation", type="Attribute")
    G.add_node("InvestorSeats", type="Attribute")
    G.add_node("FounderSeats", type="Attribute")

    # Define Relationships (Edges)
    G.add_edge("LiquidationPreference", "Multiple", relation="hasAttribute")
    G.add_edge("LiquidationPreference", "Participation", relation="hasAttribute")
    G.add_edge("BoardOfDirectors", "InvestorSeats", relation="hasAttribute")
    G.add_edge("BoardOfDirectors", "FounderSeats", relation="hasAttribute")

    # Define Market Norms citing Source Docs
    G.add_node("1x_Non_Participating", type="MarketNorm")
    G.add_edge("1x_Non_Participating", "NVCA_Model_Term_Sheet", relation="citedIn")

    return G
