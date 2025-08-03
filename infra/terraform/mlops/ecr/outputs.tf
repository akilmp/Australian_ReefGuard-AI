output "repository_url" {
  description = "URL of the ECR repository"
  value       = aws_ecr_repository.this.repository_url
}

output "arn" {
  description = "ARN of the repository"
  value       = aws_ecr_repository.this.arn
}
