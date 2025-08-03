variable "name" {
  description = "Name prefix for the VPC and subnets"
  type        = string
}

variable "cidr_block" {
  description = "CIDR block for the VPC"
  type        = string
}

variable "public_subnet_cidrs" {
  description = "Map of public subnet CIDR blocks"
  type        = map(string)
}

variable "private_subnet_cidrs" {
  description = "Map of private subnet CIDR blocks"
  type        = map(string)
}

variable "tags" {
  description = "Additional tags to apply"
  type        = map(string)
  default     = {}
}
