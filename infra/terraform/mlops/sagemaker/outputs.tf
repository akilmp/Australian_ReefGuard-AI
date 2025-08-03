output "notebook_arn" {
  description = "ARN of the notebook instance"
  value       = aws_sagemaker_notebook_instance.this.arn
}
