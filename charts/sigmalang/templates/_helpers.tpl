{{/*
Expand the name of the chart.
*/}}
{{- define "sigmalang.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "sigmalang.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "sigmalang.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "sigmalang.labels" -}}
helm.sh/chart: {{ include "sigmalang.chart" . }}
{{ include "sigmalang.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "sigmalang.selectorLabels" -}}
app.kubernetes.io/name: {{ include "sigmalang.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: api
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "sigmalang.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "sigmalang.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Return the proper namespace
*/}}
{{- define "sigmalang.namespace" -}}
{{- if .Values.namespaceOverride -}}
{{- .Values.namespaceOverride -}}
{{- else -}}
{{- .Release.Namespace -}}
{{- end -}}
{{- end -}}

{{/*
Return the Redis host
*/}}
{{- define "sigmalang.redis.host" -}}
{{- if .Values.redis.enabled -}}
{{- printf "%s-redis-master" (include "sigmalang.fullname" .) -}}
{{- else -}}
{{- .Values.externalRedis.host -}}
{{- end -}}
{{- end -}}

{{/*
Return the Redis port
*/}}
{{- define "sigmalang.redis.port" -}}
{{- if .Values.redis.enabled -}}
6379
{{- else -}}
{{- .Values.externalRedis.port | default "6379" -}}
{{- end -}}
{{- end -}}
