output "log_group_arn" {
  description = "ARN of the log group"
  value       = aws_cloudwatch_log_group.this.arn
}
