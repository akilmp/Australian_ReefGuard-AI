variable "name" {
  description = "Name of the notebook instance"
  type        = string
}

variable "role_arn" {
  description = "Execution role for the notebook"
  type        = string
}

variable "instance_type" {
  description = "SageMaker notebook instance type"
  type        = string
  default     = "ml.t3.medium"
}

variable "tags" {
  description = "Additional tags"
  type        = map(string)
  default     = {}
}
