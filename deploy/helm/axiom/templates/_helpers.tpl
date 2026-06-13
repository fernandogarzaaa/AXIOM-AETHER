{{/* Expand the name of the chart. */}}
{{- define "axiom.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Fully qualified app name. */}}
{{- define "axiom.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "axiom.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Common labels. */}}
{{- define "axiom.labels" -}}
helm.sh/chart: {{ include "axiom.chart" . }}
{{ include "axiom.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/* Selector labels. */}}
{{- define "axiom.selectorLabels" -}}
app.kubernetes.io/name: {{ include "axiom.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/* Headless service name. */}}
{{- define "axiom.headlessName" -}}
{{- printf "%s-headless" (include "axiom.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/* Name of the Secret to use (existing or chart-created). */}}
{{- define "axiom.secretName" -}}
{{- if .Values.secrets.existingSecret -}}
{{- .Values.secrets.existingSecret -}}
{{- else -}}
{{- include "axiom.fullname" . -}}
{{- end -}}
{{- end -}}

{{/* ServiceAccount name for the gossip CronJob. */}}
{{- define "axiom.gossipServiceAccountName" -}}
{{- printf "%s-gossip" (include "axiom.fullname" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}
