variable "name" {
  description = "Name of the CloudWatch log group"
  type        = string
}

variable "retention_in_days" {
  description = "Retention period for logs"
  type        = number
  default     = 14
}

variable "tags" {
  description = "Additional tags"
  type        = map(string)
  default     = {}
}
