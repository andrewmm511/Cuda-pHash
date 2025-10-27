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
    
    log "Deleting ${VM_NAME}*"

    az vm delete \
        --resource-group "$RESOURCE_GROUP" \
        --name "$VM_NAME" \
        --force-deletion true
        --yes
    
    log "Cleanup complete!"
}

main