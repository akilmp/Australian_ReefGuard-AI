terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

resource "aws_sagemaker_notebook_instance" "this" {
  name          = var.name
  role_arn      = var.role_arn
  instance_type = var.instance_type

  tags = var.tags
}
