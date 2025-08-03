variable "name" {
  description = "Bucket name"
  type        = string
}

variable "acl" {
  description = "Canned ACL to apply"
  type        = string
  default     = "private"
}

variable "tags" {
  description = "Additional tags"
  type        = map(string)
  default     = {}
}
