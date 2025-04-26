#!/bin/bash

# Ensure that the script runs in non-interactive mode
export DEBIAN_FRONTEND=noninteractive

# Update the package lists
sudo apt-get update -y

# Install Docker
sudo apt-get install -y docker.io

# Start and enable Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Install necessary utilities
sudo apt-get install -y unzip curl

# Download and install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/home/ubuntu/awscliv2.zip"
unzip -o /home/ubuntu/awscliv2.zip -d /home/ubuntu/
sudo /home/ubuntu/aws/install

# Add 'ubuntu' user to the 'docker' group to run Docker commands without 'sudo'
sudo usermod -aG docker ubuntu

# Clean up the AWS CLI installation files
rm -rf /home/ubuntu/awscliv2.zip /home/ubuntu/aws

# ===== CODEDEPLOY AGENT FIX SECTION ==
echo "===== FIXING CODEDEPLOY AGENT ISSUES ====="

# 1. Check if CodeDeploy agent is installed and reinstall if needed
if [ ! -d "/opt/codedeploy-agent" ] || [ ! -f "/opt/codedeploy-agent/bin/codedeploy-agent" ]; then
  echo "CodeDeploy agent not found or incomplete. Reinstalling..."
  
  # Remove any existing installation
  sudo service codedeploy-agent stop 2>/dev/null
  sudo apt-get remove -y codedeploy-agent 2>/dev/null
  sudo rm -rf /opt/codedeploy-agent 2>/dev/null
  
  # Install dependencies
  sudo apt-get install -y ruby-full wget
  
  # Get the region from the instance 
  REGION=$(curl -s http://169.254.169.254/latest/dynamic/instance-identity/document | grep region | awk -F" '{print $4}')
  if [ -z "$REGION" ]; then
    # Default to us-east-1 if we can't determine the region
    REGION="us-east-1"
  fi
  
  # Download and install the CodeDeploy agent
  cd /home/ubuntu
  wget https://aws-codedeploy-$REGION.s3.amazonaws.com/latest/install
  chmod +x ./install
  sudo ./install auto
  rm -f ./install
  
  echo "CodeDeploy agent reinstalled."
else
  echo "CodeDeploy agent installation found."
fi

# 2. Fix permissions and ownership
echo "Fixing CodeDeploy agent permissions..."
sudo chown -R root:root /opt/codedeploy-agent
sudo chmod -R 755 /opt/codedeploy-agent

# 3. Clear the deployment cache
echo "Clearing CodeDeploy deployment cache..."
sudo rm -rf /opt/codedeploy-agent/deployment-root/deployment-instructions/*
sudo rm -rf /opt/codedeploy-agent/deployment-root/deployment-logs/*
sudo rm -rf /opt/codedeploy-agent/deployment-root/*/d-*

# 4. Ensure the CodeDeploy agent service is properly configured
echo "Ensuring CodeDeploy agent service is properly configured..."
if [ -f "/etc/init.d/codedeploy-agent" ]; then
  sudo /etc/init.d/codedeploy-agent stop
  sudo /etc/init.d/codedeploy-agent start
elif [ -f "/etc/systemd/system/codedeploy-agent.service" ]; then
  sudo systemctl daemon-reload
  sudo systemctl stop codedeploy-agent
  sudo systemctl start codedeploy-agent
  sudo systemctl enable codedeploy-agent
else
  echo "Creating systemd service for CodeDeploy agent..."
  cat << 'EOF' | sudo tee /etc/systemd/system/codedeploy-agent.service
[Unit]
Description=AWS CodeDeploy Host Agent
After=network.target

[Service]
Type=simple
ExecStart=/opt/codedeploy-agent/bin/codedeploy-agent start
ExecStop=/opt/codedeploy-agent/bin/codedeploy-agent stop
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload
  sudo systemctl start codedeploy-agent
  sudo systemctl enable codedeploy-agent
fi

# 5. Verify network connectivity to CodeDeploy service
echo "Verifying network connectivity to CodeDeploy service..."
REGION=$(curl -s http://169.254.169.254/latest/dynamic/instance-identity/document | grep region | awk -F" '{print $4}')
if curl -s https://codedeploy.$REGION.amazonaws.com > /dev/null; then
  echo "Network connectivity to CodeDeploy service is working."
else
  echo "WARNING: Cannot connect to CodeDeploy service. Check security groups and network ACLs."
fi

# 6. Restart the agent and verify it's running
echo "Restarting CodeDeploy agent..."
sudo service codedeploy-agent restart
sleep 5

if sudo service codedeploy-agent status | grep -q "running"; then
  echo "SUCCESS: CodeDeploy agent is now running."
else
  echo "ERROR: CodeDeploy agent is still not running. Checking logs..."
  sudo cat /var/log/aws/codedeploy-agent/codedeploy-agent.log | tail -n 20
fi

# 7. Check for common error patterns in the logs
echo "Checking for common error patterns in logs..."
if sudo grep -q "InstanceAgentException" /var/log/aws/codedeploy-agent/codedeploy-agent.log; then
  echo "Found InstanceAgentException in logs. This often indicates permission issues."
  echo "Ensure the instance has an IAM role with AmazonEC2RoleforAWSCodeDeploy policy."
fi

if sudo grep -q "Unable to access the endpoint" /var/log/aws/codedeploy-agent/codedeploy-agent.log; then
  echo "Found endpoint access issues in logs. This indicates network connectivity problems."
  echo "Check security groups, network ACLs, and internet connectivity."
fi

if sudo grep -q "Missing credentials" /var/log/aws/codedeploy-agent/codedeploy-agent.log; then
  echo "Found missing credentials in logs. This indicates IAM role issues."
  echo "Ensure the instance has an IAM role with proper permissions."
fi

echo "===== CODEDEPLOY AGENT FIX COMPLETE ====="