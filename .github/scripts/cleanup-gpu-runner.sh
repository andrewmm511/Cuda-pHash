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

check_vm_exists() {
    log "Checking if VM exists..."
    
    VM_EXISTS=$(az vm show \
        --resource-group "$RESOURCE_GROUP" \
        --name "$VM_NAME" \
        --query name -o tsv 2>/dev/null || echo "")
    
    if [ -z "$VM_EXISTS" ]; then
        log "VM '$VM_NAME' not found. May have already been deleted."
        return 1
    fi
    
    log "VM exists: $VM_NAME"
    return 0
}

delete_vm() {
    log "Deleting VM: $VM_NAME"
    
    az vm delete \
        --resource-group "$RESOURCE_GROUP" \
        --name "$VM_NAME" \
        --yes \
        --no-wait
    
    log "VM deletion initiated (async)"
}

delete_network_resources() {
    log "Cleaning up network resources..."
    
    log "Deleting Network Interface..."
    az network nic delete \
        --resource-group "$RESOURCE_GROUP" \
        --name "${VM_NAME}-nic" \
        --no-wait 2>/dev/null || log "NIC not found or already deleted"
    
    log "Deleting Public IP..."
    az network public-ip delete \
        --resource-group "$RESOURCE_GROUP" \
        --name "${VM_NAME}-ip" \
        --no-wait 2>/dev/null || log "Public IP not found or already deleted"
    
    log "Deleting Network Security Group..."
    az network nsg delete \
        --resource-group "$RESOURCE_GROUP" \
        --name "${VM_NAME}-nsg" \
        --no-wait 2>/dev/null || log "NSG not found or already deleted"
    
    log "Deleting Virtual Network..."
    az network vnet delete \
        --resource-group "$RESOURCE_GROUP" \
        --name "${VM_NAME}-vnet" \
        --no-wait 2>/dev/null || log "VNet not found or already deleted"
    
    log "Network resources cleanup initiated (async)"
}

delete_disks() {
    log "Cleaning up managed disks..."
    
    DISKS=$(az disk list \
        --resource-group "$RESOURCE_GROUP" \
        --query "[?tags.github_run_id!=null && contains(name, '$VM_NAME')].name" \
        -o tsv 2>/dev/null || echo "")
    
    if [ -n "$DISKS" ]; then
        echo "$DISKS" | while read -r disk; do
            if [ -n "$disk" ]; then
                log "Deleting disk: $disk"
                az disk delete \
                    --resource-group "$RESOURCE_GROUP" \
                    --name "$disk" \
                    --yes \
                    --no-wait 2>/dev/null || log "Failed to delete disk: $disk"
            fi
        done
    else
        log "No disks found for cleanup"
    fi
}

verify_cleanup() {
    log "Verifying cleanup..."
    
    sleep 5
    
    VM_EXISTS=$(az vm show \
        --resource-group "$RESOURCE_GROUP" \
        --name "$VM_NAME" \
        --query name -o tsv 2>/dev/null || echo "")
    
    if [ -z "$VM_EXISTS" ]; then
        log "VM deletion confirmed"
    else
        log "VM still exists (deletion in progress)"
    fi
    
    REMAINING=$(az resource list \
        --resource-group "$RESOURCE_GROUP" \
        --tag "github_run_id=${GITHUB_RUN_ID}" \
        --query "length(@)" -o tsv 2>/dev/null || echo "0")
    
    log "Remaining resources for this run: $REMAINING"
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    log "=========================================="
    log "Azure GPU Runner Cleanup Script"
    log "=========================================="
    log "Resource Group: $RESOURCE_GROUP"
    log "VM Name: $VM_NAME"
    log "=========================================="
    
    if ! check_vm_exists; then
        log "No cleanup needed - VM doesn't exist"
        exit 0
    fi
    
    delete_vm
    
    delete_network_resources
    
    delete_disks
    
    verify_cleanup
    
    log "=========================================="
    log "Cleanup Initiated!"
    log "Note: Deletions are asynchronous and may"
    log "take a few minutes to complete fully."
    log "=========================================="
}

main