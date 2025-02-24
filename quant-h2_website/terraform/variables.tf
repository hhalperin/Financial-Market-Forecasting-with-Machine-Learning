########################################################################
# variables.tf
########################################################################
variable "domain_name" {
  type        = string
  description = "The root domain name to set up (e.g. quant-h2.com)."
  default     = "quant-h2.com"
}

variable "environment" {
  type        = string
  description = "Environment name (e.g. Production, Staging)."
  default     = "Production"
}

variable "enable_website_uploads" {
  type        = bool
  description = "Control whether to upload local files to S3."
  default     = true
}
