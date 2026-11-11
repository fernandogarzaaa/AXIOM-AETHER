#include <stdio.h>
#include <string.h>

#define LEN 4

int main (int argc, char* *argv)
{
	const char check[LEN] = {'H', 'Q', '9'};
	char p[101];
	int flag = 1; // FALSE
	
	scanf("%s", &p);
	
	for (int i = 0; i < LEN; i++)
		if (strchr(p, check[i]) != NULL)
		{
			flag = 0;	// TRUE
			break;
		}
		
	if (flag == 0)
		printf("YES");
	else
		printf("NO");
	
	return 0;
}