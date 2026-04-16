export const classHierarchy = {
  name: 'Class Hierarchy',
  description: '8 classes with extends, implements, uses edges',
  script: `CREATE NODE "Animal" kind = "class" name = "Animal" abstract = "true"
CREATE NODE "Dog" kind = "class" name = "Dog"
CREATE NODE "Cat" kind = "class" name = "Cat"
CREATE NODE "Serializable" kind = "class" name = "Serializable" abstract = "true"
CREATE NODE "Comparable" kind = "class" name = "Comparable" abstract = "true"
CREATE NODE "PetDog" kind = "class" name = "PetDog"
CREATE NODE "DogFood" kind = "class" name = "DogFood"
CREATE NODE "PetShop" kind = "class" name = "PetShop"
CREATE EDGE "Dog" -> "Animal" kind = "extends"
CREATE EDGE "Cat" -> "Animal" kind = "extends"
CREATE EDGE "PetDog" -> "Dog" kind = "extends"
CREATE EDGE "Dog" -> "Serializable" kind = "implements"
CREATE EDGE "Cat" -> "Serializable" kind = "implements"
CREATE EDGE "Dog" -> "Comparable" kind = "implements"
CREATE EDGE "PetShop" -> "PetDog" kind = "uses"
CREATE EDGE "PetShop" -> "DogFood" kind = "uses"

// Try these queries:
// ANCESTORS OF "PetDog" DEPTH 3
// DESCENDANTS OF "Animal" DEPTH 2
// MATCH ("PetDog") -[kind = "extends"]-> (parent) -[kind = "extends"]-> (grandparent)
// NODES WHERE INDEGREE > 1`,
}
