variable "name" {
  description = "Name of the EKS cluster"
  type        = string
}

variable "role_arn" {
  description = "IAM role ARN for the EKS cluster"
  type        = string
}

variable "subnet_ids" {
  description = "Subnets associated with the cluster"
  type        = list(string)
}

variable "cluster_version" {
  description = "Kubernetes version for the cluster"
  type        = string
  default     = "1.29"
}

variable "tags" {
  description = "Additional tags to apply"
  type        = map(string)
  default     = {}
}
