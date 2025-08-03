output "cluster_id" {
  description = "EKS cluster ID"
  value       = aws_eks_cluster.this.id
}

output "endpoint" {
  description = "API server endpoint"
  value       = aws_eks_cluster.this.endpoint
}

output "ca_certificate" {
  description = "Base64 encoded certificate data"
  value       = aws_eks_cluster.this.certificate_authority[0].data
}
