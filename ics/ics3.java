

import java.math.BigInteger; 
import java.security.MessageDigest; 
import java.security.NoSuchAlgorithmException; 

public class SHA1{ 
	public static String encryptThisString(String input) 
	{ 
		try { 
			
			MessageDigest md = MessageDigest.getInstance("SHAAA-1"); 

			
			byte[] messageDigest = md.digest(input.getBytes()); 

			 
			BigInteger no = new BigInteger(1, messageDigest); 
 
			String hashtext = no.toString(16); 

			
			while (hashtext.length() < 32) { 
				hashtext = "0" + hashtext; 
			} 

		
			return hashtext; 
		} 

		
		catch (NoSuchAlgorithmException e) { 
			throw new RuntimeException(e); 
		} 
	} 

	
	public static void main(String args[]) throws
									NoSuchAlgorithmException 
	{ 

		System.out.println("HashCode Generated by SHA-1 for: "); 

		String s1 = "Hello world"; 
		System.out.println("\n" + s1 + " : " + encryptThisString(s1)); 

		String s2 = "hello world"; 
		System.out.println("\n" + s2 + " : " + encryptThisString(s2)); 
	} 
} 
/*
OUTPUT:-
C:\Users\lenovo\Desktop\New folder (3)>javac SHA1.java
C:\Users\lenovo\Desktop\New folder (3)>java SHA1
HashCode Generated by SHA-1 for:

Hello world : 7b502c3a1f48c8609ae212cdfb639dee39673f5e

hello world : 2aae6c35c94fcfb415dbe95f408b9ce91ee846ed

C:\Users\lenovo\Desktop\New folder (3)>
*/
