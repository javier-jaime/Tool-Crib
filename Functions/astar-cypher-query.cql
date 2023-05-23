// Import manhattan road network graphml (from github and drop into graph import folder) 

CALL apoc.import.graphml("manhattan_road_network.graphml", {defaultRelationshipType:"LINKED_TO"})

// Create Graph Projection

CALL gds.graph.project.cypher("road_network",         
"MATCH (n:Junction) RETURN id(n) as id, n.latitude as latitude, n.longitude as longitude",         
"MATCH (n)-[r:LINKED_TO]->(m) RETURN id(n) as source, id(m) as target, toInteger(r.length) as length")

// Run the Shorthest Path algorithm and stream results

MATCH (source:Junction {osmid: "42428297"})
MATCH (target:Junction {osmid: "42423674"})
CALL gds.shortestPath.astar.stream('road_network', {
    sourceNode: source,
    targetNode: target,
    latitudeProperty: 'latitude',
    longitudeProperty: 'longitude',
    relationshipWeightProperty: 'length'
})
YIELD index, sourceNode, targetNode, totalCost, nodeIds, costs, path
RETURN index, sourceNode, targetNode, totalCost, 
    [nodeId IN nodeIds | gds.util.asNode(nodeId).osmid] AS Osmids,
    costs,
    nodes(path) as path
ORDER BY index