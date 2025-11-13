#!/bin/sh

# -----------------------------------------------------------------------------
# Logging

__secrets_manager_log() {
  echo "$(date +"%Y-%m-%dT%H:%M:%S.%3N") - $1 - $2" >&2
}

__secrets_manager_info() {
  __secrets_manager_log "INFO" "$1"
}

__secrets_manager_warn() {
  __secrets_manager_log "WARN" "$1"
}

__secrets_manager_error() {
  __secrets_manager_log "ERROR" "$1"
}

# -----------------------------------------------------------------------------
# Read from secretsmanager and export as env var

__secrets_manager_export() {
  __secrets_manager_info "Fetching secret from $2"

  __secrets_manager_secret=$(
    aws secretsmanager get-secret-value --secret-id "$2" \
      --query "SecretString" --output text 2> /dev/null
  )

  if [ -z "${__secrets_manager_secret}" ]; then
    __secrets_manager_warn "Cannot fetch secret $2"
    __secrets_manager_warn "Ensure you have access to the secret and it's value is set"
    unset __secrets_manager_secret
    return 1
  fi

  export "$1"="$__secrets_manager_secret"

  unset __secrets_manager_secret
}

# -----------------------------------------------------------------------------
# Load all secrets

__secrets_manager() {

  # Ensure AWS cli is available
  if ! which aws > /dev/null 2>&1; then
      __secrets_manager_error "Cannot find aws command"
      __secrets_manager_error "Please ensure the awscli package is installed"
      return 1
  fi

  # Get email part of the current assumed role, if this fails it is highly
  # likely that the AWS session is not set or has expired, or some other type
  # of assumed role is using this script that we are yet to account for.
  __secrets_manager_aws_arn_email=$(
    aws sts get-caller-identity --output text 2> /dev/null \
      | awk '{ print $2 }'                                 \
      | rev                                                \
      | cut -f 1 -d /                                      \
      | rev
  )

  if [ -z "${__secrets_manager_aws_arn_email}" ]; then
    __secrets_manager_error "Cannot find name in AWS role ARN"
    __secrets_manager_error "Ensure your AWS environment variables are exported for a valid session"
    unset __secrets_manager_aws_arn_email
    return 1
  fi

  __secrets_manager_user=$(echo "$__secrets_manager_aws_arn_email" | cut -f1 -d@)

  # Fetch each secret and export as correctly named env var
  __secrets_manager_info "Fetching secrets for $__secrets_manager_aws_arn_email"

  __secrets_manager_export HF_TOKEN \
    "ai-research/$__secrets_manager_user/hf-token"

  __secrets_manager_export OPENAI_API_KEY \
    "ai-research/global/azure-openai-api-key"

  unset __secrets_manager_user
  unset __secrets_manager_aws_arn_email

  return 0
}

__secrets_manager

unset __secrets_manager_log
unset __secrets_manager_info
unset __secrets_manager_warn
unset __secrets_manager_error
unset __secrets_manager_export
unset __secrets_manager
