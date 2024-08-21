from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from .forms import StudentRegistrationForm, ProfessorRegistrationForm, StudentLoginForm, ProfessorLoginForm

@login_required
def user_logout(request):
    logout(request)
    return redirect('home')

def student_register(request):
    if request.method == 'POST':
        form = StudentRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect_to_fastapi(request, user, 'generate_response-ui')
    else:
        form = StudentRegistrationForm()
    return render(request, 'student_register.html', {'form': form})

def professor_register(request):
    if request.method == 'POST':
        form = ProfessorRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect_to_fastapi(request, user, 'upload-pdf-ui')
    else:
        form = ProfessorRegistrationForm()
    return render(request, 'professor_register.html', {'form': form})

def student_login(request):
    if request.method == 'POST':
        form = StudentLoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None and user.is_student:
                login(request, user)
                return redirect_to_fastapi(request, user, 'generate-response-ui')
    else:
        form = StudentLoginForm()
    return render(request, 'student_login.html', {'form': form})

def professor_login(request):
    if request.method == 'POST':
        form = ProfessorLoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None and user.is_teacher:
                login(request, user)
                return redirect_to_fastapi(request, user, 'upload_pdf-ui')
    else:
        form = ProfessorLoginForm()
    return render(request, 'professor_login.html', {'form': form})

def home(request):
    return render(request, 'home.html')

def redirect_to_fastapi(request, user, endpoint):
    fastapi_base_url = "http://127.0.0.1:5000"  # Replace with your actual FastAPI server URL
    response = redirect(f"{fastapi_base_url}/{endpoint}")  # Redirect to the appropriate FastAPI endpoint
    response.set_cookie('sessionid', request.COOKIES.get('sessionid'))
    return response
