#!/bin/bash
# cleanup-gpu-runner.sh
# Deletes Azure GPU VM and all associated resources
set -e

# ============================================================================
# Configuration
# ============================================================================

RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-gh-runners-rg}"
SHORT_RUN_ID="${GITHUB_RUN_ID: -6}"
VM_NAME="${VM_NAME:-ghr-${SHORT_RUN_ID}-${GITHUB_RUN_ATTEMPT}}"

# ============================================================================
# Functions
# ============================================================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

main() {
    log "=========================================="
    log "Azure GPU Runner Cleanup"
    log "Resource Group: $RESOURCE_GROUP"
    log "VM Name: $VM_NAME"
    log "=========================================="
    
    if ! az vm show -g "$RESOURCE_GROUP" -n "$VM_NAME" &>/dev/null; then
        log "VM '$VM_NAME' not found. Nothing to clean up."
        exit 0
    fi
    
    log "Finding all resources with pattern: ${VM_NAME}*"
    
    RESOURCE_IDS=$(az resource list --resource-group "$RESOURCE_GROUP" --query "[?starts_with(name, '$VM_NAME')].id" -o tsv)
    
    if [ -n "$RESOURCE_IDS" ]; then
        log "Found $(echo "$RESOURCE_IDS" | wc -l) resources to delete"
        
        echo "$RESOURCE_IDS" | xargs -P 10 -I {} az resource delete --ids {} --no-wait 2>/dev/null || true
        
        log "Deletion initiated for all resources"
    else
        log "No resources found matching pattern"
    fi
    
    log "Cleanup complete!"
}

main