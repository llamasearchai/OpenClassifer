#!/bin/bash
# =============================================================================
# OpenClassifier Production Deployment Script
# =============================================================================

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"
ENV_FILE="$PROJECT_ROOT/.env"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker is not running. Please start Docker daemon."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if .env file exists
    if [[ ! -f "$ENV_FILE" ]]; then
        log_warning ".env file not found. Creating from template..."
        if [[ -f "$PROJECT_ROOT/env.example" ]]; then
            cp "$PROJECT_ROOT/env.example" "$ENV_FILE"
            log_warning "Please update the .env file with your configuration before running again."
            exit 1
        else
            log_error "env.example template not found. Cannot create .env file."
            exit 1
        fi
    fi
    
    # Check if required environment variables are set
    source "$ENV_FILE"
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        log_error "OPENAI_API_KEY is not set in .env file."
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create required directories
setup_directories() {
    log_info "Setting up directories..."
    
    directories=(
        "$PROJECT_ROOT/logs"
        "$PROJECT_ROOT/model_cache"
        "$PROJECT_ROOT/monitoring/grafana/dashboards"
        "$PROJECT_ROOT/monitoring/grafana/datasources"
        "$PROJECT_ROOT/monitoring/logstash/config"
        "$PROJECT_ROOT/monitoring/logstash/pipeline"
        "$PROJECT_ROOT/nginx"
    )
    
    for dir in "${directories[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        fi
    done
    
    log_success "Directories setup complete"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build the main application image
    docker-compose build --no-cache openclassifier
    
    log_success "Docker images built successfully"
}

# Deploy services
deploy_services() {
    log_info "Deploying services..."
    
    cd "$PROJECT_ROOT"
    
    # Deploy core services first
    log_info "Starting core services (Redis, Prometheus)..."
    docker-compose up -d redis prometheus
    
    # Wait for core services to be ready
    log_info "Waiting for core services to be ready..."
    sleep 10
    
    # Deploy main application
    log_info "Starting OpenClassifier application..."
    docker-compose up -d openclassifier
    
    # Wait for main application to be ready
    log_info "Waiting for application to be ready..."
    sleep 20
    
    # Deploy replica and other services
    log_info "Starting replica and monitoring services..."
    docker-compose up -d openclassifier_replica grafana nginx
    
    # Deploy optional services
    log_info "Starting optional services..."
    docker-compose up -d elasticsearch kibana logstash redis_exporter node_exporter
    
    log_success "All services deployed"
}

# Health check
health_check() {
    log_info "Performing health checks..."
    
    # Check main application
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s http://localhost:8000/health > /dev/null; then
            log_success "Main application is healthy"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            log_error "Main application health check failed after $max_attempts attempts"
            return 1
        fi
        
        log_info "Health check attempt $attempt/$max_attempts failed, retrying in 5 seconds..."
        sleep 5
        ((attempt++))
    done
    
    # Check replica
    if curl -f -s http://localhost:8001/health > /dev/null; then
        log_success "Replica application is healthy"
    else
        log_warning "Replica application health check failed"
    fi
    
    # Check Redis
    if docker-compose exec -T redis redis-cli ping | grep -q PONG; then
        log_success "Redis is healthy"
    else
        log_warning "Redis health check failed"
    fi
    
    # Check Prometheus
    if curl -f -s http://localhost:9090/-/healthy > /dev/null; then
        log_success "Prometheus is healthy"
    else
        log_warning "Prometheus health check failed"
    fi
    
    # Check Grafana
    if curl -f -s http://localhost:3000/api/health > /dev/null; then
        log_success "Grafana is healthy"
    else
        log_warning "Grafana health check failed"
    fi
    
    log_success "Health checks completed"
}

# Show deployment status
show_status() {
    log_info "Deployment Status:"
    echo "=================="
    
    cd "$PROJECT_ROOT"
    docker-compose ps
    
    echo ""
    log_info "Service URLs:"
    echo "  Main Application:    http://localhost:8000"
    echo "  Replica Application: http://localhost:8001"
    echo "  API Documentation:   http://localhost:8000/docs"
    echo "  Prometheus:          http://localhost:9090"
    echo "  Grafana:            http://localhost:3000"
    echo "  Kibana:             http://localhost:5601"
    echo "  Redis:              localhost:6379"
    
    echo ""
    log_info "Health Endpoints:"
    echo "  Application Health:  http://localhost:8000/health"
    echo "  Application Metrics: http://localhost:8000/metrics"
    
    echo ""
    log_info "Default Credentials:"
    echo "  Grafana: admin / admin123 (change after first login)"
}

# Cleanup function
cleanup() {
    log_info "Stopping all services..."
    cd "$PROJECT_ROOT"
    docker-compose down --remove-orphans
}

# Rollback function
rollback() {
    log_warning "Rolling back deployment..."
    cleanup
    log_success "Rollback completed"
}

# Main deployment function
main() {
    log_info "Starting OpenClassifier deployment..."
    
    # Set trap for cleanup on script exit
    trap cleanup EXIT
    
    # Parse command line arguments
    case "${1:-deploy}" in
        "deploy")
            check_prerequisites
            setup_directories
            build_images
            deploy_services
            health_check
            show_status
            
            # Remove trap as deployment succeeded
            trap - EXIT
            
            log_success "Deployment completed successfully!"
            ;;
        "stop")
            cleanup
            log_success "All services stopped"
            ;;
        "restart")
            cleanup
            sleep 5
            main deploy
            ;;
        "status")
            show_status
            ;;
        "logs")
            cd "$PROJECT_ROOT"
            docker-compose logs -f "${2:-openclassifier}"
            ;;
        "health")
            health_check
            ;;
        "rollback")
            rollback
            ;;
        *)
            echo "Usage: $0 [deploy|stop|restart|status|logs|health|rollback]"
            echo ""
            echo "Commands:"
            echo "  deploy   - Deploy all services (default)"
            echo "  stop     - Stop all services"
            echo "  restart  - Restart all services"
            echo "  status   - Show deployment status"
            echo "  logs     - Show logs (specify service name as second argument)"
            echo "  health   - Run health checks"
            echo "  rollback - Rollback deployment"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@" 