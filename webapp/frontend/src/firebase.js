import { initializeApp } from "firebase/app";
import { getAuth, GoogleAuthProvider, signInWithPopup } from "firebase/auth";

const firebaseConfig = {
  apiKey: "AIzaSy.................................",
  authDomain: "aviss-webapp.firebaseapp.com",
  projectId: "aviss-webapp",
  storageBucket: "aviss-webapp.firebasestorage.app",
  messagingSenderId: "6687275942",
  appId: "1:6687275942:web:......................"
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const googleProvider = new GoogleAuthProvider();
googleProvider.setCustomParameters({
  prompt: 'select_account'
});
export { signInWithPopup };
