import java.io.*; 

class CRT { 
	
	
// Function to find modulo inverse of a
static int  modInverse(int a, int m)
{
    IntWrapper x = new IntWrapper(0);
      IntWrapper y = new IntWrapper(0);
    int g = gcdExtended(a, m, x, y);
    if (g != 1){
        System.out.println( "Inverse doesn't exist");
	return 0;
    }
    else
    {
        // m is added to handle negative x
        int res = (x.a%m + m) % m;
        System.out.println("Modular multiplicative inverse is "+res);
        return res;
    }
}

// C function for extended Euclidean Algorithm
static int gcdExtended(int a, int b, IntWrapper x, IntWrapper y)
{
    // Base Case
    if (a == 0)
    {
        x.a = 0; y.a = 1;
        return b;
    }

    IntWrapper x1=new IntWrapper(0) ; 
	IntWrapper y1=new IntWrapper(0);
    int gcd = gcdExtended(b%a, a, x1, y1);

    // Update x and y using results of recursive
    // call
    x.a = y1.a - (b/a) * x1.a;
    y.a = x1.a;

    return gcd;
}


static int findMinX(int num[], int rem[], int k)
{
    // Compute product of all numbers
    int prod = 1;
    for (int i = 0; i < k; i++)
        prod *= num[i];

    // Initialize result
    int result = 0;

    // Apply above formula
    for (int i = 0; i < k; i++)
    {
        int pp = prod / num[i];
        result += rem[i] * modInverse(pp, num[i]) * pp;
    }

    return result % prod;
}
	
	
	public static void main(String args[]) 
	{ 
		int num[] = {3, 4, 5}; 
		int rem[] = {2, 3, 1}; 
		int k = num.length; 
		System.out.println("x is " +findMinX(num, rem, k)); 
	} 
} 
class IntWrapper {
   public int a;
   public IntWrapper(int a){ this.a = a;}
}
/*
OUTPUT:-
C:\Users\lenovo\Desktop\New folder (3)>javac CRT.java

C:\Users\lenovo\Desktop\New folder (3)>java CRT
x is 11

*/
