#!/bin/bash
# provision-gpu-runner.sh
# Creates an Azure GPU VM from a custom image for GitHub Actions self-hosted runner
set -e

# ============================================================================
# Configuration
# ============================================================================

RESOURCE_GROUP="${AZURE_RESOURCE_GROUP:-gh-runners-rg}"
LOCATION="${AZURE_LOCATION:-eastus}"
IMAGE_NAME="${IMAGE_NAME:-gh-runner-image}"
VM_NAME="${VM_NAME:-ghr-vm}"
VM_SIZE="${VM_SIZE:-Standard_NV6ads_A10_v5}"
ADMIN_USERNAME="${ADMIN_USERNAME:-azureuser}"
ADMIN_PASSWORD="${ADMIN_PASSWORD}"

# ============================================================================
# Functions
# ============================================================================

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

create_vm() {
    local image_id=$1
    
    log "Creating VM: $VM_NAME"
    log "Size: $VM_SIZE"
    log "Location: $LOCATION"
    
    az vm create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$VM_NAME" \
        --image "$IMAGE_NAME" \
        --size "$VM_SIZE" \
        --location "$LOCATION" \
        --admin-username "$ADMIN_USERNAME" \
        --admin-password "$ADMIN_PASSWORD" \
        --public-ip-address "${VM_NAME}-ip" \
        --public-ip-sku Standard \
        --nsg "${VM_NAME}-nsg" \
        --storage-sku Premium_LRS \
        --os-disk-size-gb 30 \
        --priority Spot \
        --max-price "1" \
        --eviction-policy Delete \
        --accelerated-networking true \
        --tags github_run_id="${GITHUB_RUN_ID:-manual}" github_repository="${GITHUB_REPOSITORY:-manual}" auto_cleanup=true \
        --output none
    
    if [ $? -eq 0 ]; then
        log "VM created successfully"
    else
        log "ERROR: Failed to create VM"
        exit 1
    fi
}

configure_nsg() {
    log "Configuring Network Security Group..."
    
    # Allow HTTPS outbound (for GitHub)
    az network nsg rule create \
        --resource-group "$RESOURCE_GROUP" \
        --nsg-name "${VM_NAME}-nsg" \
        --name AllowHTTPS \
        --priority 1001 \
        --direction Outbound \
        --access Allow \
        --protocol Tcp \
        --source-address-prefixes '*' \
        --source-port-ranges '*' \
        --destination-address-prefixes '*' \
        --destination-port-ranges 443 \
        --output none 2>/dev/null || true
    
    log "NSG configured"
}

get_vm_info() {
    log "Retrieving VM information..."
    
    VM_IP=$(az vm show -d \
        --resource-group "$RESOURCE_GROUP" \
        --name "$VM_NAME" \
        --query publicIps -o tsv)
    
    VM_ID=$(az vm show \
        --resource-group "$RESOURCE_GROUP" \
        --name "$VM_NAME" \
        --query id -o tsv)
    
    log "VM Public IP: $VM_IP"
    log "VM ID: $VM_ID"
}

# ============================================================================
# Main Execution
# ============================================================================

main() {
    log "=========================================="
    log "Azure GPU Runner Provisioning Script"
    log "=========================================="
    log "Resource Group: $RESOURCE_GROUP"
    log "VM Name: $VM_NAME"
    log "Location: $LOCATION"
    log "=========================================="

    create_vm
    
    configure_nsg
    
    get_vm_info
    
    if [ -n "$GITHUB_OUTPUT" ]; then
        log "Writing outputs to GitHub Actions..."
        echo "vm_name=$VM_NAME" >> "$GITHUB_OUTPUT"
        echo "vm_id=$VM_ID" >> "$GITHUB_OUTPUT"
        echo "vm_ip=$VM_IP" >> "$GITHUB_OUTPUT"
    fi
    
    log "=========================================="
    log "Provisioning Complete!"
    log "VM Name: $VM_NAME"
    log "=========================================="
}

main