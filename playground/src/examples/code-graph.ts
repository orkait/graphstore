export const codeGraph = {
  name: 'Code Graph',
  description: '~20 nodes: functions, classes, modules with multiple edge types',
  script: `BEGIN
CREATE NODE "mod_auth" kind = "module" name = "auth" file = "src/auth/"
CREATE NODE "mod_api" kind = "module" name = "api" file = "src/api/"
CREATE NODE "mod_db" kind = "module" name = "db" file = "src/db/"
CREATE NODE "cls_User" kind = "class" name = "User" file = "src/auth/models.py"
CREATE NODE "cls_Session" kind = "class" name = "Session" file = "src/auth/session.py"
CREATE NODE "cls_Token" kind = "class" name = "Token" file = "src/auth/token.py"
CREATE NODE "cls_BaseModel" kind = "class" name = "BaseModel" file = "src/db/base.py"
CREATE NODE "cls_Router" kind = "class" name = "Router" file = "src/api/router.py"
CREATE NODE "fn_login" kind = "function" name = "login" file = "src/auth/handlers.py"
CREATE NODE "fn_logout" kind = "function" name = "logout" file = "src/auth/handlers.py"
CREATE NODE "fn_verify" kind = "function" name = "verify_token" file = "src/auth/token.py"
CREATE NODE "fn_hash" kind = "function" name = "hash_password" file = "src/auth/crypto.py"
CREATE NODE "fn_query" kind = "function" name = "query" file = "src/db/engine.py"
CREATE NODE "fn_connect" kind = "function" name = "connect" file = "src/db/engine.py"
CREATE NODE "fn_handle_req" kind = "function" name = "handle_request" file = "src/api/handler.py"
CREATE NODE "fn_serialize" kind = "function" name = "serialize" file = "src/api/serial.py"
COMMIT

CREATE EDGE "mod_auth" -> "cls_User" kind = "contains"
CREATE EDGE "mod_auth" -> "cls_Session" kind = "contains"
CREATE EDGE "mod_auth" -> "cls_Token" kind = "contains"
CREATE EDGE "mod_db" -> "cls_BaseModel" kind = "contains"
CREATE EDGE "mod_api" -> "cls_Router" kind = "contains"
CREATE EDGE "cls_User" -> "cls_BaseModel" kind = "extends"
CREATE EDGE "cls_Session" -> "cls_BaseModel" kind = "extends"
CREATE EDGE "fn_login" -> "fn_verify" kind = "calls"
CREATE EDGE "fn_login" -> "fn_hash" kind = "calls"
CREATE EDGE "fn_login" -> "fn_query" kind = "calls"
CREATE EDGE "fn_logout" -> "fn_query" kind = "calls"
CREATE EDGE "fn_handle_req" -> "fn_login" kind = "calls"
CREATE EDGE "fn_handle_req" -> "fn_verify" kind = "calls"
CREATE EDGE "fn_handle_req" -> "fn_serialize" kind = "calls"
CREATE EDGE "fn_query" -> "fn_connect" kind = "calls"
CREATE EDGE "mod_api" -> "mod_auth" kind = "imports"
CREATE EDGE "mod_auth" -> "mod_db" kind = "imports"

// Try these queries:
// SUBGRAPH FROM "fn_login" DEPTH 2
// TRAVERSE FROM "fn_handle_req" DEPTH 3 WHERE kind = "calls"
// MATCH ("fn_handle_req") -[kind = "calls"]-> (a) -[kind = "calls"]-> (b)
// SYS STATS`,
}
