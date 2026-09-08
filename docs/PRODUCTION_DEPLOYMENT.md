# Production Deployment

This deployment uses Cloud Build, Artifact Registry, an IAP-protected Cloud Run
service with one NVIDIA L4 GPU, and Cloudflare Workers. Private adapters and router artifacts are copied
from GCS into the container image by Cloud Build. The Cloud Run container never
needs the Google Cloud CLI.

## Where to run the commands

Run `gcloud builds submit` from the Git repository root: the directory that
contains `cloudbuild.yaml`, `.github/`, `symbolic-moe/`, and `web app/`.

```powershell
Set-Location "E:\PROFILE\Projects\Skill-Based-Expert-Routing-System-For-Multilabel-Hate-Speech-Detection-main\Skill-Based-Expert-Routing-System-For-Multilabel-Hate-Speech-Detection-main"
Test-Path .\cloudbuild.yaml
```

The test must print `True`. `gcloud run deploy` only consumes an already-built
image and can technically run from any directory, but running both commands
from the repository root avoids using the wrong build context.

PowerShell commands below use a backtick as the final character for line
continuation. Do not add characters after it, and do not paste Markdown escapes
such as `\_`, `\@`, or `[url](url)` into the shell.

## Runtime architecture

- A Cloudflare Worker serves the React/Vite static assets.
- Cloud Run's direct IAP integration protects every ingress path, including the
  `run.app` URL, without requiring an external load balancer.
- One L4 GPU instance loads a shared Llama 3.1 model and four QLoRA adapters.
- Cloud Build stages the base model, adapters, profiles, and skills from
  `gs://serml-app-artifacts/artifacts` before building the image.
- The service scales to zero and is capped at one instance because the project
  currently has one non-zonally-redundant L4 GPU of regional quota.

The GitHub workflow discovers the IAP-protected `run.app` URL and injects it as
`VITE_API_BASE_URL`. A browser user must establish an IAP session before using
the API from the Cloudflare origin.

## One-time Google Cloud setup

Enable the required APIs:

```powershell
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com iap.googleapis.com cloudresourcemanager.googleapis.com serviceusage.googleapis.com --project=serml-app
```

Create the runtime service account and Artifact Registry repository if they do
not already exist:

```powershell
gcloud iam service-accounts describe serml-api@serml-app.iam.gserviceaccount.com --project=serml-app
gcloud artifacts repositories describe serml-repo --location=us-central1 --project=serml-app
```

If either lookup returns `NOT_FOUND`, create only that missing resource:

```powershell
gcloud iam service-accounts create serml-api --display-name="SERML API Backend" --project=serml-app
gcloud artifacts repositories create serml-repo --repository-format=docker --location=us-central1 --project=serml-app
```

Create `hf-token` once, or add a new secret version when rotating it. Avoid
placing the real token in committed files.

```powershell
$hfToken = "hf_replace_with_your_token"
$tokenFile = New-TemporaryFile
Set-Content -LiteralPath $tokenFile -Value $hfToken -NoNewline

gcloud secrets describe hf-token --project=serml-app
# If the secret does not exist:
gcloud secrets create hf-token --replication-policy=automatic --data-file=$tokenFile --project=serml-app
# If the secret already exists, use this instead:
gcloud secrets versions add hf-token --data-file=$tokenFile --project=serml-app

Remove-Item -LiteralPath $tokenFile
$hfToken = $null
```

The runtime service account does not read Hugging Face or GCS at runtime; all
model files are baked into the image. Create a dedicated, user-specified build
identity so builds never execute as the broadly permissioned default Compute
Engine account. Give it steady-state read access to the source and artifact
buckets, write access to the Docker repository, and Cloud Logging access:

```powershell
$projectId = "serml-app"
$buildServiceAccount = "serml-build@$projectId.iam.gserviceaccount.com"
$buildMember = "serviceAccount:$buildServiceAccount"

gcloud iam service-accounts describe $buildServiceAccount --project=$projectId
# If the lookup returns NOT_FOUND, create it once:
gcloud iam service-accounts create serml-build `
  --display-name="SERML Cloud Build executor" `
  --project=$projectId

gcloud storage buckets add-iam-policy-binding gs://serml-app-artifacts `
  --member=$buildMember `
  --role="roles/storage.objectViewer"
gcloud storage buckets add-iam-policy-binding gs://serml-app_cloudbuild `
  --member=$buildMember `
  --role="roles/storage.objectViewer"
gcloud storage buckets add-iam-policy-binding gs://serml-app_cloudbuild `
  --member=$buildMember `
  --role="roles/storage.bucketViewer"
gcloud artifacts repositories add-iam-policy-binding serml-repo `
  --location=us-central1 `
  --project=$projectId `
  --member=$buildMember `
  --role="roles/artifactregistry.writer"
gcloud projects add-iam-policy-binding $projectId `
  --member=$buildMember `
  --role="roles/logging.logWriter" `
  --condition=None
gcloud projects add-iam-policy-binding $projectId `
  --member=$buildMember `
  --role="roles/serviceusage.serviceUsageConsumer" `
  --condition=None
```

GitHub authenticates without a service-account key. Create a dedicated deploy
identity and a Workload Identity provider restricted to this repository's
immutable GitHub IDs, the `main` branch, and the production workflow:

```powershell
$projectId = "serml-app"
$projectNumber = "437199970190"
$poolId = "github-actions"
$providerId = "github-main"
$deploySa = "serml-deploy@$projectId.iam.gserviceaccount.com"
$deployMember = "serviceAccount:$deploySa"
$repo = "muditbaid/Skill-Based-Expert-Routing-System-For-Multilabel-Hate-Speech-Detection"
$repoId = "1082669769"
$ownerId = "88234346"

gcloud services enable iam.googleapis.com iamcredentials.googleapis.com `
  sts.googleapis.com --project=$projectId
gcloud iam service-accounts create serml-deploy `
  --display-name="SERML GitHub production deployer" `
  --project=$projectId
gcloud iam workload-identity-pools create $poolId `
  --location=global `
  --display-name="GitHub Actions" `
  --description="GitHub Actions identities for SERML" `
  --project=$projectId

$mapping = "google.subject=assertion.sub,attribute.repository_id=assertion.repository_id,attribute.repository_owner_id=assertion.repository_owner_id,attribute.ref=assertion.ref,attribute.workflow_ref=assertion.workflow_ref"
$condition = "assertion.repository_id == '$repoId' && assertion.repository_owner_id == '$ownerId' && assertion.ref == 'refs/heads/main' && assertion.workflow_ref == '$repo/.github/workflows/deploy.yml@refs/heads/main'"

gcloud iam workload-identity-pools providers create-oidc $providerId `
  --location=global `
  --workload-identity-pool=$poolId `
  --issuer-uri="https://token.actions.githubusercontent.com/" `
  --attribute-mapping=$mapping `
  --attribute-condition=$condition `
  --display-name="SERML main deploy workflow" `
  --project=$projectId

$principalSet = "principalSet://iam.googleapis.com/projects/$projectNumber/locations/global/workloadIdentityPools/$poolId/attribute.repository_id/$repoId"
gcloud iam service-accounts add-iam-policy-binding $deploySa `
  --member=$principalSet `
  --role="roles/iam.workloadIdentityUser" `
  --project=$projectId
```

Grant only the permissions used by the workflow:

```powershell
$projectRoles = @(
  "roles/cloudbuild.builds.editor",
  "roles/run.admin",
  "roles/logging.viewer",
  "roles/serviceusage.serviceUsageConsumer"
)
foreach ($role in $projectRoles) {
  gcloud projects add-iam-policy-binding $projectId `
    --member=$deployMember `
    --role=$role `
    --condition=None
}

gcloud storage buckets add-iam-policy-binding gs://serml-app_cloudbuild `
  --member=$deployMember `
  --role="roles/storage.bucketViewer"
gcloud storage buckets add-iam-policy-binding gs://serml-app_cloudbuild `
  --member=$deployMember `
  --role="roles/storage.objectCreator"
gcloud artifacts repositories add-iam-policy-binding serml-repo `
  --location=us-central1 `
  --project=$projectId `
  --member=$deployMember `
  --role="roles/artifactregistry.reader"
gcloud iam service-accounts add-iam-policy-binding `
  serml-api@$projectId.iam.gserviceaccount.com `
  --project=$projectId `
  --member=$deployMember `
  --role="roles/iam.serviceAccountUser"
gcloud iam service-accounts add-iam-policy-binding `
  serml-build@$projectId.iam.gserviceaccount.com `
  --project=$projectId `
  --member=$deployMember `
  --role="roles/iam.serviceAccountUser"
```

Enable direct IAP once on the existing service. This creates/configures the IAP
service agent and disables public invocation:

```powershell
gcloud run services update serml-api `
  --project=serml-app `
  --region=us-central1 `
  --iap `
  --invoker-iam-check `
  --no-allow-unauthenticated
```

For a personal project or users outside a Google organization, enable direct
IAP once from **Cloud Run → serml-api → Security → Require authentication →
Identity-Aware Proxy (IAP)** and configure an External OAuth audience. This
first OAuth-client setup cannot always be completed entirely through `gcloud`.
The production workflow deliberately does not receive IAP Policy Admin or OAuth
Config Editor. Complete this one-time IAP setup as a project administrator; the
workflow verifies that IAP remains enabled and only maintains the Cloud Run
invoker binding for the IAP service agent.

## Required GCS artifacts

`cloudbuild.yaml` requires these objects:

```text
gs://serml-app-artifacts/artifacts/
├── saves/llama31-8b/dynahate/qlora/{adapter_config.json,adapter_model.safetensors}
├── saves/llama31-8b/tweeteval_offensive/qlora/{adapter_config.json,adapter_model.safetensors}
├── saves/llama31-8b/kaggle_cyberbullying/qlora/{adapter_config.json,adapter_model.safetensors}
├── saves/llama31-8b/jigsaw_threat/qlora/{adapter_config.json,adapter_model.safetensors}
└── symbolic-moe/
    ├── profiles.json
    └── skills.txt
```

Only the two required files for each adapter are staged; checkpoints, trainer
state, tokenizer copies, and `.git` metadata are not added to the image.

The base model is required under the immutable revision prefix
`gs://serml-app-artifacts/artifacts/base-model/0e9e39f249a16976918f6564b8830bc894c89659/`.
It must contain a Hugging Face
`config.json`, `tokenizer_config.json`, all model shards, and an empty
`.complete` marker uploaded only after every model file. Cloud Build fails fast
when this cache is incomplete so a cold instance never has to download the
model before Cloud Run's startup deadline.

Stage the pinned base model directly from Hugging Face to GCS. The token is
available only to the download step, and both temporary grants are removed even
if the build fails:

```powershell
$projectId = "serml-app"
$buildServiceAccount = "serml-build@$projectId.iam.gserviceaccount.com"
$buildMember = "serviceAccount:$buildServiceAccount"

gcloud secrets add-iam-policy-binding hf-token `
  --project=$projectId `
  --member=$buildMember `
  --role="roles/secretmanager.secretAccessor"
gcloud storage buckets add-iam-policy-binding gs://serml-app-artifacts `
  --member=$buildMember `
  --role="roles/storage.objectAdmin"

try {
  gcloud builds submit . `
    --config=cloudbuild.stage-base-model.yaml `
    --project=$projectId
  if ($LASTEXITCODE -ne 0) { throw "Base-model staging build failed" }
} finally {
  gcloud secrets remove-iam-policy-binding hf-token `
    --project=$projectId `
    --member=$buildMember `
    --role="roles/secretmanager.secretAccessor"
  gcloud storage buckets remove-iam-policy-binding gs://serml-app-artifacts `
    --member=$buildMember `
    --role="roles/storage.objectAdmin"
}

gcloud storage ls gs://serml-app-artifacts/artifacts/base-model/0e9e39f249a16976918f6564b8830bc894c89659/.complete
gcloud storage ls gs://serml-app-artifacts/artifacts/base-model/0e9e39f249a16976918f6564b8830bc894c89659/manifest.json
```

The staging build pins model revision
`0e9e39f249a16976918f6564b8830bc894c89659`, validates every shard referenced
by `model.safetensors.index.json`, records and verifies SHA-256 for every model
and tokenizer file, and uploads `.complete` last. A finalized revision is never
overwritten; use a new revision prefix when intentionally upgrading the model.

## Build the backend image

From the repository root:

```powershell
$image = "us-central1-docker.pkg.dev/serml-app/serml-repo/serml-api:$(git rev-parse --short HEAD)"

gcloud builds submit . `
  --config=cloudbuild.yaml `
  --substitutions="_IMAGE=$image" `
  --project=serml-app
```

Cloud Build stages the private artifacts, installs the pinned CUDA/PyTorch
stack, verifies that the active interpreter has `Python.h`, and pushes the
immutable image tag.

## Deploy Cloud Run

Use the `$image` value from the build step:

```powershell
gcloud run deploy serml-api `
  --project=serml-app `
  --region=us-central1 `
  --image=$image `
  --execution-environment=gen2 `
  --gpu=1 `
  --gpu-type=nvidia-l4 `
  --no-gpu-zonal-redundancy `
  --cpu=8 `
  --memory=32Gi `
  --cpu-boost `
  --no-cpu-throttling `
  --min=0 `
  --max=1 `
  --min-instances=0 `
  --max-instances=1 `
  --concurrency=1 `
  --timeout=900 `
  --port=8000 `
  --startup-probe="httpGet.path=/health/ready,httpGet.port=8000,initialDelaySeconds=0,periodSeconds=10,timeoutSeconds=5,failureThreshold=24" `
  --set-env-vars="MODEL_BACKEND=local_inprocess,SYMBOLIC_MOE_DIR=/app/symbolic-moe,MODEL_PRELOAD=true,MODEL_WARMUP=true,BASE_MODEL_PATH=/app/artifacts/base-model,REQUIRE_CUDA=true,APP_ENV=prod,ALLOW_ORIGINS=https://sberhsd.muditb0712.workers.dev" `
  --remove-secrets=HF_TOKEN `
  --service-account="serml-api@serml-app.iam.gserviceaccount.com" `
  --ingress=all `
  --invoker-iam-check `
  --iap `
  --no-allow-unauthenticated
```

Grant the IAP service agent permission to invoke the service, permit CORS
preflight at IAP, and grant a user access:

```powershell
$projectNumber = gcloud projects describe serml-app --format="value(projectNumber)"
$userEmail = "muditb0712@gmail.com"

gcloud run services add-iam-policy-binding serml-api `
  --project=serml-app `
  --region=us-central1 `
  --member="serviceAccount:service-$projectNumber@gcp-sa-iap.iam.gserviceaccount.com" `
  --role="roles/run.invoker"

gcloud iap settings set iap-settings.yaml `
  --project=serml-app `
  --resource-type=cloud-run `
  --region=us-central1 `
  --service=serml-api

gcloud iap web add-iam-policy-binding `
  --project=serml-app `
  --resource-type=cloud-run `
  --region=us-central1 `
  --service=serml-api `
  --member="user:$userEmail" `
  --role="roles/iap.httpsResourceAccessor"
```

`/health/live` confirms that the web process is alive. `/health/ready` returns
HTTP 503 while the base model and adapters load and a short end-to-end warmup
runs across all four adapters. It returns HTTP 200 only after the CUDA/Triton
inference path succeeds.
The Cloud Run startup probe uses this readiness endpoint, so a failed warmup
marks the new revision unhealthy and prevents it from receiving production
traffic.

## Frontend and CI/CD

The GitHub workflow in `.github/workflows/deploy.yml` builds the backend through
Cloud Build, deploys Cloud Run, waits for the model-preload completion log,
builds the frontend with the discovered service URL, and deploys the existing
Cloudflare Worker with Wrangler.

Google Cloud authentication uses branch-restricted Workload Identity
Federation, so no long-lived Google service-account key is stored in GitHub.
Configure these GitHub repository secrets:

- `CLOUDFLARE_API_TOKEN`: token permitted to deploy the `sberhsd` Worker.
- `CLOUDFLARE_ACCOUNT_ID`: Cloudflare account ID.

After the first successful federated deployment, delete the obsolete
`GCP_SA_KEY` repository secret in **GitHub -> Settings -> Secrets and variables
-> Actions**. If that JSON key still exists on its service account, identify its
`client_email` and `private_key_id` from your original secure copy and delete
that exact key with `gcloud iam service-accounts keys delete`; do not guess a key
ID or remove unrelated keys.

Open the deployed frontend and select **Connect protected API**. Complete the
Google/IAP sign-in in the new tab, then return to the frontend and analyze a
sample. The frontend sends cross-origin cookies with API requests, and
`iap-settings.yaml` allows the unauthenticated CORS preflight while IAP still
protects the actual request.

## Local development

Install PyTorch from the CUDA lane separately, matching the Docker build:

```powershell
Set-Location "web app/backend"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

In a second PowerShell terminal with Node.js 22 or newer:

```powershell
Set-Location "web app/frontend"
npm ci
Copy-Item .env.example .env
# Set VITE_API_BASE_URL=http://127.0.0.1:8000 in .env
npm run dev
```
