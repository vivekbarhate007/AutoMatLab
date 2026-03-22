**🚀 AutoAgentOps**
Autonomous DevOps + Observability System with Agentic AI
AutoAgentOps is a full-stack, agent-driven observability platform designed to simulate how real-world monitoring systems (like Datadog or PagerDuty) work internally. It combines real-time incident detection, distributed tracing, idempotent systems, and chaos engineering into a single interactive system.



**🔍 Overview**
AutoAgentOps demonstrates how modern distributed systems handle failures and monitoring:
Detects latency spikes and system anomalies automatically
Uses correlation IDs for end-to-end request tracing
Implements idempotency for exactly-once execution
Applies RSA-256 JWT authentication for secure communication
Simulates failures using a built-in chaos engineering suite
Provides a live dashboard for real-time monitoring and debugging



| Layer        | Component                  | Description                                                                |
|--------------|---------------------------|-----------------------------------------------------------------------------|
| Frontend     | React Dashboard           | UI for monitoring, orders, audit logs, and demo interactions                |
| API Layer    | REST API (JWT RS256)      | Secure communication using RSA-256 JWT authentication                       |
| Backend      | Node.js / Express         | Core application logic and request handling                                 |
| Auth Service | RSA-256 JWT               | Public key-based authentication and token verification                      |
| Order Service| Idempotency Engine        | Ensures exactly-once execution using Idempotency-Key pattern                |
| Observability| Detection Engine          | Computes P95 latency, error rates, and auto-triggers incidents              |
| Tracing      | Correlation System        | Tracks requests end-to-end (HTTP → DB)                                      |
| Chaos Suite  | Failure Injection         | Simulates latency and errors for system resilience testing                  |
| Database     | SQLite                    | Stores audit_events, orders, and users                                      |



**⚡ Key Features**
**🔹 Real-time Incident Detection**
Background engine continuously monitors system metrics (P95 latency, error rates) and automatically triggers incidents when thresholds are exceeded.
**🔹 Distributed Tracing**
Each request carries a correlationId, enabling full visibility from API request to database operation.
**🔹 Exactly-once Semantics**
Prevents duplicate operations using an Idempotency-Key pattern, ensuring system consistency.
**🔹 Secure Authentication**
Implements RSA-256 JWT authentication with public/private key separation.
**🔹 Chaos Engineering**
Simulate real-world failures (latency spikes, errors) and observe system behavior in real time.



**🛠️ Tech Stack**
**Layer	Technologies**
Frontend	React, TypeScript, Tailwind CSS, Recharts
Backend	Node.js, Express
Auth	JWT RS256, Node Crypto
Database	SQLite
Infra	Docker (optional), UUID
Tooling	Vite



**🚀 Getting Started**
1. Clone Repository
git clone https://github.com/vivekbarhate007/IncidentIQ.git
cd IncidentIQ

3. Install Dependencies
npm install

5. Configure Environment
cp .env.example .env

7. Run Application
npm run dev
Open 👉 http://localhost:3000



🎯 Demo Walkthrough
**Scenario 1 — Detect Incident**
Inject latency using Chaos Suite
Observe spike detection
Incident auto-created

**Scenario 2 — Trace Request**
Place an order
Track correlationId across system
View full request lifecycle

**Scenario 3 — Validate Security**
Decode JWT token
Verify RS256 usage
Confirm public-key-based verification



**📁 Project Structure**
| Layer      | Subsystem            | Component                     | Description                                      |
|------------|---------------------|-------------------------------|--------------------------------------------------|
| Frontend   | UI Layer            | Dashboard                     | Displays system metrics and incidents            |
|            |                     | Orders                        | Manage and create orders                         |
|            |                     | Audit Logs                    | View request traces and logs                     |
|            |                     | Demo Guide                    | Step-by-step system walkthrough                  |
|------------|---------------------|-------------------------------|--------------------------------------------------|
| API Layer  | Communication       | REST API (JWT RS256)          | Secure request handling and authentication       |
|------------|---------------------|-------------------------------|--------------------------------------------------|
| Backend    | Core Services       | Auth Service                  | RSA-256 JWT authentication & verification        |
|            |                     | Order Service                 | Idempotent order processing                      |
|            |                     | Observability Engine          | Detects latency spikes and error rates           |
|            |                     | Correlation Tracing           | Tracks requests (HTTP → DB)                      |
|            |                     | Chaos Suite                   | Injects failures for testing                     |
|------------|---------------------|-------------------------------|--------------------------------------------------|
| Observability | Metrics Engine   | P95 Latency Calculation       | Measures system performance                      |
|            |                     | Error Detection               | Identifies anomalies and failures                |
|------------|---------------------|-------------------------------|--------------------------------------------------|
| Chaos      | Failure Injection   | Inject Latency                | Simulates slow system behavior                   |
|            |                     | Inject Errors                 | Simulates system failures                        |
|------------|---------------------|-------------------------------|--------------------------------------------------|
| Database   | Storage Layer       | audit_events                  | Stores logs and system events                    |
|            |                     | orders                        | Stores order data                                |
|            |                     | users                         | Stores user data                                 |



**🧠 How It Works**
**Observability Engine**
Continuously polls audit_events
Computes latency and error rates
Triggers incidents automatically
**Correlation System**
Assigns unique IDs to each request
Tracks request flow across services
**Idempotency System**
Ensures safe retries
Prevents duplicate execution



**⚙️ Design Decisions**
**SQLite over Postgres:** Simple local setup, easily replaceable
**RSA-256 over HS256: **Secure separation of signing & verification
**Polling Engine: **Lightweight alternative to Kafka-based streaming



**🔮 Future Improvements**
Integrate event streaming (Kafka / Redis)
Add distributed microservices architecture
Extend to cloud-native deployment (AWS/GCP)
Incorporate AI-driven anomaly prediction



**👨‍💻 Author**
**Vivek Barhate**
MS Computer Science — George Mason University (2026)
Interested in Full-Stack Engineering, Distributed Systems, and AI/ML
