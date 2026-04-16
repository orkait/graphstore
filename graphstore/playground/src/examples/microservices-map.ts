export const microservicesMap = {
  name: 'Microservices Map',
  description: 'Services, databases, queues with API calls and messaging',
  script: `BEGIN
CREATE NODE "api_gateway" kind = "service" name = "API Gateway" port = 8080
CREATE NODE "auth_svc" kind = "service" name = "Auth Service" port = 8081
CREATE NODE "user_svc" kind = "service" name = "User Service" port = 8082
CREATE NODE "order_svc" kind = "service" name = "Order Service" port = 8083
CREATE NODE "payment_svc" kind = "service" name = "Payment Service" port = 8084
CREATE NODE "notify_svc" kind = "service" name = "Notification Service" port = 8085
CREATE NODE "users_db" kind = "database" name = "Users DB" engine = "postgres"
CREATE NODE "orders_db" kind = "database" name = "Orders DB" engine = "postgres"
CREATE NODE "payments_db" kind = "database" name = "Payments DB" engine = "postgres"
CREATE NODE "event_bus" kind = "queue" name = "Event Bus" engine = "kafka"
CREATE NODE "email_queue" kind = "queue" name = "Email Queue" engine = "rabbitmq"
COMMIT

CREATE EDGE "api_gateway" -> "auth_svc" kind = "calls_api"
CREATE EDGE "api_gateway" -> "user_svc" kind = "calls_api"
CREATE EDGE "api_gateway" -> "order_svc" kind = "calls_api"
CREATE EDGE "auth_svc" -> "users_db" kind = "reads_from"
CREATE EDGE "user_svc" -> "users_db" kind = "reads_from"
CREATE EDGE "order_svc" -> "orders_db" kind = "reads_from"
CREATE EDGE "payment_svc" -> "payments_db" kind = "reads_from"
CREATE EDGE "order_svc" -> "payment_svc" kind = "calls_api"
CREATE EDGE "order_svc" -> "event_bus" kind = "publishes_to"
CREATE EDGE "payment_svc" -> "event_bus" kind = "publishes_to"
CREATE EDGE "notify_svc" -> "event_bus" kind = "subscribes_to"
CREATE EDGE "notify_svc" -> "email_queue" kind = "publishes_to"

// Try these queries:
// PATH FROM "api_gateway" TO "payments_db" MAX_DEPTH 5
// DISTANCE FROM "api_gateway" TO "notify_svc" MAX_DEPTH 5
// NODES WHERE kind = "service"
// EDGES FROM "api_gateway" WHERE kind = "calls_api"
// DESCENDANTS OF "api_gateway" DEPTH 3`,
}
