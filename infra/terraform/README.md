# FinCloud AWS deployment

Terraform deploys the React/Vite site to the existing `fincloud-market-data-unique123` S3 bucket and provisions a serverless API in `ap-south-1`. The bucket itself and pre-existing research objects are not Terraform managed.

## Managed resources

- S3 website configuration, exact root website object policy, bucket public-access-block configuration, and the generated site objects.
- Lambda deployment package object under `lambda/` (SSE-S3).
- API Gateway HTTP API, default stage, Lambda proxy integration, nine API routes, and Lambda invoke permission.
- Python 3.12 Lambda function, scoped execution role/policy, and seven-day CloudWatch log group.

The API reads only required Silver data, selected model artifacts and research summary keys. It writes analytical results only to `gold/api-results/*`. Website users can read only the exact root HTML and built asset objects. The six research prefixes (`bronze`, `silver`, `gold`, `models`, `experiments`, `metadata`) remain private to anonymous S3 requests. S3 website hosting is HTTP-only; the website does not use CloudFront or provide HTTPS. The API is publicly callable, with CORS restricted to the website origin and API Gateway throttling enabled; it has no end-user authentication.

## Build and deploy

Build the Lambda zip using `backend/build_lambda_package.ps1`. Build the frontend with the deployed API URL configured:

```powershell
$env:VITE_API_BASE_URL = (terraform output -raw api_url)
cd ../../frontend
npm test
npm run build
Remove-Item Env:VITE_API_BASE_URL
cd ../infra/terraform
terraform fmt -check
terraform validate
terraform plan -out fincloud.tfplan
terraform show fincloud.tfplan
terraform apply fincloud.tfplan
```

Before applying, verify the plan does not create or replace the bucket and does not change unrelated S3 prefixes. The generated plan and state files may contain infrastructure metadata; keep them private.

## Data and model uploads

`backend/upload_research_data.ps1` and `backend/upload_model_artifacts.ps1` default to dry run. They refuse overwrites, preserve unrelated keys, use SSE-S3, and verify uploaded size and checksum. Apply the API role's narrow S3 permissions through Terraform; never make research prefixes public.

## Cleanup

Review `terraform plan -destroy` before running `terraform destroy`. This removes the managed API, role/policy, log group, package object, website objects/configuration, website policy and bucket public access block resource, but does not delete the bucket or uploaded research objects. If removing the website, restore bucket-level Block Public Access to all `true` flags after reviewing any other bucket users. API request and compute charges are usage based; review AWS billing separately.
