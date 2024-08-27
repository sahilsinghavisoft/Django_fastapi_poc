from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from .forms import UserLoginForm, UserRegistrationForm

@login_required
def user_logout(request):
    logout(request)
    return redirect('home')

def user_register(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            # Redirect to a FastAPI endpoint after registration
            return redirect_to_fastapi(request, 'dashboard')
    else:
        form = UserRegistrationForm()
    
    return render(request, 'user_register.html', {'form': form})

def user_login(request):
    if request.method == 'POST':
        form = UserLoginForm(request.POST)
        print("got the form")
        if form.is_valid():
            print("form is valid")
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                print("got the user")
                login(request, user)
                if user:
                    print("login successful")
                    return redirect_to_fastapi(request, 'dashboard')
    else:
        form = UserLoginForm()
    return render(request, 'user_login.html', {'form': form})

def home(request):
    return render(request, 'home.html')

def redirect_to_fastapi(request, endpoint):
    fastapi_base_url = "http://127.0.0.1:5000"  # Replace with your actual FastAPI server URL
    response = redirect(f"{fastapi_base_url}/{endpoint}")  # Redirect to the appropriate FastAPI endpoint
    response.set_cookie('sessionid', request.COOKIES.get('sessionid'))
    return response
