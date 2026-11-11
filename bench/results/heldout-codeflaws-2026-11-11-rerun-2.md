# Held-out validation of Codeflaws repair-test passes

Source table: `repair-unseen-codeflaws-rerun-2-2026-11-11.md` (7 repair-test passes). Held-out inputs: `heldout-input-*` / `heldout-output-*` shipped pinned in each fixture, never run by the repair oracle. Same compile flags and stdout normalization as `verify.sh`; each run bounded at 10 s.

**Held-out validated: 4 of 7 repair-test passes also pass every held-out input.**

| Fixture | Held-out (solver patch) | Held-out (reference fix) | Verdict |
|---|---|---|---|
| 131-A-bug-18069181-18073244 | 55/55 | 55/55 | HELDOUT-PASS |
| 133-A-bug-18048780-18048793 | 84/84 | 84/84 | HELDOUT-PASS |
| 136-A-bug-17499279-17499383 | 100/100 | 100/100 | HELDOUT-PASS |
| 231-A-bug-18055428-18055539 | 7/10 (failed: heldout-input-pos20, heldout-input-pos7, heldout-input-pos8) | 10/10 | HELDOUT-FAIL |
| 467-A-bug-18348808-18348843 | 24/24 | 24/24 | HELDOUT-PASS |
| 58-A-bug-17762757-17762804 | 25/39 (failed: heldout-input-pos12, heldout-input-pos15, heldout-input-pos18, heldout-input-pos25 (exit -6), heldout-input-pos26 (exit -6), heldout-input-pos27 (exit -6), heldout-input-pos28 (exit -6), heldout-input-pos29 (exit -6), heldout-input-pos30 (exit -6), heldout-input-pos32, heldout-input-pos33 (exit -6), heldout-input-pos34 (exit -6), heldout-input-pos38, heldout-input-pos40 (exit -6)) | 30/39 (failed: heldout-input-pos25 (exit -6), heldout-input-pos26 (exit -6), heldout-input-pos27 (exit -6), heldout-input-pos28 (exit -6), heldout-input-pos29 (exit -6), heldout-input-pos30 (exit -6), heldout-input-pos33 (exit -6), heldout-input-pos34 (exit -6), heldout-input-pos40 (exit -6)) | REFERENCE-INCONSISTENT |
| 96-A-bug-18284844-18284870 | 21/43 (failed: heldout-input-pos10, heldout-input-pos11, heldout-input-pos14, heldout-input-pos15, heldout-input-pos16, heldout-input-pos17, heldout-input-pos18, heldout-input-pos19, heldout-input-pos21, heldout-input-pos22, heldout-input-pos24, heldout-input-pos25, heldout-input-pos27, heldout-input-pos28, heldout-input-pos29, heldout-input-pos30, heldout-input-pos32, heldout-input-pos35, heldout-input-pos39, heldout-input-pos40, heldout-input-pos6, heldout-input-pos8) | 43/43 | HELDOUT-FAIL |

## Solver accepted edits (repaired .c vs buggy .c)

### 131-A-bug-18069181-18073244 (HELDOUT-PASS)

```diff
--- 
+++ 
@@ -18,7 +18,7 @@
 		printf("%s",str);}
 	else
 		if(y==0){
-		for(i=1;str[i];i++){
+		for(i=0;str[i];i++){
 		str[i]=tolower(str[i]);}
 		printf("%s",str);}
 		
```

### 133-A-bug-18048780-18048793 (HELDOUT-PASS)

```diff
--- 
+++ 
@@ -1,7 +1,7 @@
 #include <stdio.h>
 #include <string.h>
 
-#define LEN 4
+#define LEN 3
 
 int main (int argc, char* *argv)
 {
```

### 136-A-bug-17499279-17499383 (HELDOUT-PASS)

```diff
--- 
+++ 
@@ -1,7 +1,7 @@
 #include <stdio.h>
 #include <string.h>
 
-#define MXS 100*2-1+1
+#define MXS 100*3-1+1
 
 int main(int argc, char *argv[]) {
     int n, k, i;
```

### 231-A-bug-18055428-18055539 (HELDOUT-FAIL)

```diff
--- 
+++ 
@@ -6,7 +6,7 @@
     int total_problems[n];
     for(i=0,count=0;i<n;i++)
     {
-        scanf("%d%d%d",&p,&v,&t);
+        scanf("%d%d%d",&v,&p,&t);
         total_problems[i]=p*100+v*10+t;
       if(total_problems[i]==011||total_problems[i]==101||total_problems[i]==110||total_problems[i]==111)
         count++;
```

### 467-A-bug-18348808-18348843 (HELDOUT-PASS)

```diff
--- 
+++ 
@@ -8,7 +8,7 @@
         for(i=0;i<n;i++)
         {
             scanf("%d%d",&p,&q);
-            if((p+2)>=q)
+            if((p+2)<=q)
                 count++;
         }
 
```

### 58-A-bug-17762757-17762804 (REFERENCE-INCONSISTENT)

```diff
--- 
+++ 
@@ -1,7 +1,7 @@
 #include<stdio.h>
 int main(int argc, char *argv[])
 {
-    int i,j=0,count=0,c=0,e=0,f=0,g=0,d=0,k,flag=0;
+    int i,j=0,count=0,c=0,e=0,f=0,g=1,d=0,k,flag=0;
     char a[100], b[5]={"hello"};
     gets(a);
      for(i=0;a[i]!='\0';i++){
```

### 96-A-bug-18284844-18284870 (HELDOUT-FAIL)

```diff
--- 
+++ 
@@ -32,7 +32,7 @@
     char s[100];
     scanf("%s",s);
     for(i=0;i<strlen(s)-1;i++){
-        if(s[i]==s[i+1]){
+        if(s[i]==s[i]){
             count++;
         }
         else {
```
