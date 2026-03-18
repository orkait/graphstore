export const functionCallGraph = {
  name: 'Function Call Graph',
  description: '5 functions with calls edges — diamond + chain pattern',
  script: `CREATE NODE "main" kind = "function" name = "main" file = "app.py"
CREATE NODE "parse_args" kind = "function" name = "parse_args" file = "cli.py"
CREATE NODE "validate" kind = "function" name = "validate" file = "utils.py"
CREATE NODE "process" kind = "function" name = "process" file = "core.py"
CREATE NODE "output" kind = "function" name = "output" file = "io.py"
CREATE EDGE "main" -> "parse_args" kind = "calls"
CREATE EDGE "main" -> "validate" kind = "calls"
CREATE EDGE "parse_args" -> "validate" kind = "calls"
CREATE EDGE "validate" -> "process" kind = "calls"
CREATE EDGE "process" -> "output" kind = "calls"

// Try these queries:
// EDGES FROM "main" WHERE kind = "calls"
// SHORTEST PATH FROM "main" TO "output" WHERE kind = "calls"
// COMMON NEIGHBORS OF "main" AND "parse_args" WHERE kind = "calls"
// TRAVERSE FROM "main" DEPTH 3 WHERE kind = "calls"`,
}
